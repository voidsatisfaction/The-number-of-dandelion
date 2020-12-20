from typing import Tuple, List

import csv
import numpy as np
import cv2
import math

class ClusterInfo:
    def __init__(self):
        self.l = math.inf
        self.r = -math.inf
        self.u = math.inf
        self.d = -math.inf

        self.dandalion_number = 1

        self.pixel_list = []
    
    @property
    def pixel_num(self) -> int:
        return len(self.pixel_list)

    @property
    def minimum_bounding_box_area(self) -> int:
        return (self.r - self.l + 1) * (self.d - self.u + 1)

    def update_l_r_u_d_by_coordinate(
        self,
        coordinate: Tuple[int, int]
    ) -> None:
        r, c = coordinate

        self.l = min(self.l, c)
        self.r = max(self.r, c)
        self.u = min(self.u, r)
        self.d = max(self.d, r)

    def append_pixel(self, coordinate: Tuple[int, int]) -> None:
        self.pixel_list.append(coordinate)

def _dfs(
    r,
    c,
    np_array,
    visited,
    total_row,
    total_column,
    cluster_info
) -> None:
    if r < 0 or \
        c < 0 or \
        r >= total_row or \
        c >= total_column or \
        np.any(np_array[r][c]) is np.False_ or \
        visited[r][c] is True:
            return

    visited[r][c] = True

    coordinate = (r, c)
    cluster_info.update_l_r_u_d_by_coordinate(coordinate)
    cluster_info.append_pixel(coordinate)

    direction_list = [
        (-1, 0),
        (1, 0),
        (0, 1),
        (0, -1)
    ]

    for direction in direction_list:
        row_direction, column_direction = direction

        next_row = r + row_direction
        next_column = c + column_direction

        _dfs(next_row, next_column, np_array, visited, total_row, total_column, cluster_info)

def _number_of_cluster(
    np_array,
    minimum_pixel_threshold=None
) -> Tuple[int, List[dict]]:
    total_row, total_column, _ = np_array.shape

    visited = [ [ False for _ in range(total_column) ] for _ in range(total_row) ]
    cluster_info_list = []

    number = 0

    for r in range(total_row):
        for c in range(total_column):
            if np.any(np_array[r][c]) is np.True_ and visited[r][c] is False:
                cluster_info = ClusterInfo()
                _dfs(
                    r,
                    c,
                    np_array,
                    visited,
                    total_row,
                    total_column,
                    cluster_info
                )

                # TODO: add threshold here?
                if minimum_pixel_threshold is not None and cluster_info.pixel_num <= minimum_pixel_threshold:
                    continue
                number += 1
                cluster_info_list.append(cluster_info)

    return number, cluster_info_list

def get_dandalion_number(
    file_path: str,
    lower_yellow=(20, 120, 120),
    upper_yellow=(30, 255, 255),
    show_picture=False,
    minimum_pixel_threshold=None,
    pixel_num_bounding_box_area_ratio_threshold=0.7,
    suspected_overlapped_cluster_size_ratio=1.6,
    definite_overlapped_cluster_size_ratio=2.0,
) -> int:
    ROW, COLUMN = 772, 500

    original_image = cv2.imread(file_path)
    resized_original_image = cv2.resize(original_image, (ROW, COLUMN))
    
    # extract yellow regions
    hsv_image = cv2.cvtColor(
        resized_original_image,
        cv2.COLOR_BGR2HSV
    )

    mask_image = cv2.inRange(
        hsv_image,
        lower_yellow,
        upper_yellow
    )

    binary_image = cv2.bitwise_and(
        resized_original_image,
        resized_original_image,
        mask=mask_image
    )

    # yellow to binary

    # noize canceling
    kernel = np.ones((2, 2), np.uint8)

    for _ in range(10):
        noize_canceled_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)

    kernel = np.ones((3, 3), np.uint8)

    for _ in range(10):
        noize_canceled_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)

    total_cluster_num, cluster_info_list = _number_of_cluster(
        noize_canceled_image,
        minimum_pixel_threshold=minimum_pixel_threshold
    )

    sorted_cluster_info_list_by_pixel_num = sorted(
        cluster_info_list,
        key=lambda cluster_info: cluster_info.pixel_num
    )
    median_pixel_num = sorted_cluster_info_list_by_pixel_num[len(sorted_cluster_info_list_by_pixel_num)//2] \
                            .pixel_num

    def is_overlapped(
        cluster_info: ClusterInfo,
        median_pixel_num: int
    ) -> bool:
        threshold = 100
        if cluster_info.pixel_num >= max(2*median_pixel_num, threshold):
            return True

        pixel_num = cluster_info.pixel_num
        pixel_num_bounding_box_area_ratio = round(pixel_num / cluster_info.minimum_bounding_box_area * 100) / 100
        if cluster_info.pixel_num >= suspected_overlapped_cluster_size_ratio * median_pixel_num and \
            pixel_num_bounding_box_area_ratio <= pixel_num_bounding_box_area_ratio_threshold:
            
            return True

        return False


    answer = total_cluster_num
    overlapped_cluster_info_list = []
    if total_cluster_num >= 10:
        overlapped_cluster_info_list = [
            cluster_info for cluster_info in cluster_info_list
                if is_overlapped(cluster_info, median_pixel_num) is True
        ]

    for overlapped_cluster_info in overlapped_cluster_info_list:
        overlapped_cluster_num = max(round(overlapped_cluster_info.pixel_num/(1 * max(median_pixel_num, 50))), 2)
        answer += (overlapped_cluster_num - 1)

        overlapped_cluster_info.dandalion_number = overlapped_cluster_num


    if show_picture is True:
        cv2.imshow('resized_original_image', resized_original_image)
        for cluster_info in sorted_cluster_info_list_by_pixel_num:
            l = cluster_info.l
            r = cluster_info.r
            u = cluster_info.u
            d = cluster_info.d

            ul, dr = (l, u), (r, d)
            color = (0, 255, 0)
            pixel_num = cluster_info.pixel_num

            cv2.rectangle(
                resized_original_image,
                ul,
                dr,
                color
            )
            # cv2.putText(
            #     resized_original_image,
            #     str(pixel_num),
            #     ul,
            #     cv2.FONT_HERSHEY_SIMPLEX,
            #     0.4,
            #     color
            # )
            cv2.putText(
                resized_original_image,
                str(cluster_info.dandalion_number),
                ul,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color
            )

            cv2.rectangle(
                noize_canceled_image,
                ul,
                dr,
                color
            )

            if round(pixel_num / cluster_info.minimum_bounding_box_area * 100) / 100 <= 0.7 and \
                cluster_info.pixel_num >= suspected_overlapped_cluster_size_ratio * median_pixel_num and \
                cluster_info.pixel_num < definite_overlapped_cluster_size_ratio * median_pixel_num:

                cv2.putText(
                    noize_canceled_image,
                    str(round(pixel_num / cluster_info.minimum_bounding_box_area * 100) / 100),
                    ul,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    color
                )
            print(pixel_num)
        print(f'median: {median_pixel_num}')
        print(f'total cluster number: {total_cluster_num}')
        print(f'adjusted total cluster number: {answer}')

        cv2.imshow('resized_original_image_with_rect', resized_original_image)
        # cv2.imshow('binary_image', binary_image)
        cv2.imshow('noize_canceled_image', noize_canceled_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return answer

if __name__ == '__main__':
    # FILE_NAME = 'answer_without_4x4_with_threshold_with_overlap_strategy_with_area_strategy5.csv'

    # all_file_path_list = [ f'./images/test40/flower ({i}).jpg' for i in range(1, 41) ]

    # with open(FILE_NAME, 'w', newline='') as csvfile:
    #     answer_csv_writer = csv.writer(csvfile, delimiter=',')

    #     answer_csv_writer.writerow([',', 'target'])

    #     for i, file_path in enumerate(all_file_path_list):
    #         answer = get_dandalion_number(
    #             file_path,
    #             lower_yellow=(20, 140, 130),
    #             upper_yellow=(30, 255, 255),
    #             show_picture=False,
    #             minimum_pixel_threshold=10,
    #             pixel_num_bounding_box_area_ratio_threshold=0.7,
    #             suspected_overlapped_cluster_size_ratio=1.6,
    #             definite_overlapped_cluster_size_ratio=2.0
    #         )

    #         file_number = i+1
    #         answer_csv_writer.writerow([f'flower ({file_number}).jpg', answer])

    #         print(f'file number: {file_number} done')
    
    file_path = './images/test40/flower (30).jpg'
    get_dandalion_number(
        file_path,
        lower_yellow=(20, 140, 130),
        upper_yellow=(30, 255, 255),
        show_picture=True,
        minimum_pixel_threshold=10,
        pixel_num_bounding_box_area_ratio_threshold=0.7,
        suspected_overlapped_cluster_size_ratio=1.6,
        definite_overlapped_cluster_size_ratio=2.0
    )