from typing import Tuple, List
import math
import csv
import numpy as np
import cv2


'''
Cluter information of dandelion pixels
'''
class ClusterInfo:
    def __init__(self):
        self.l = math.inf
        self.r = -math.inf
        self.u = math.inf
        self.d = -math.inf

        self.dandelion_number = 1

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


'''
Dandelion counter
'''
class DandelionCounter:
    def __init__(
        self,
        lower_yellow=(20, 140, 140), # hsv
        upper_yellow=(30, 255, 255), # hsv
        minimum_pixel_threshold=8,
        pixel_num_bounding_box_area_ratio_threshold=0.7,
        suspected_overlapped_cluster_size_ratio=1.6,
    ) -> None:
        self._resize_row = 772
        self._resize_column = 500
        self._lower_yellow = lower_yellow
        self._upper_yellow = upper_yellow
        self._minimum_pixel_threshold = minimum_pixel_threshold
        self._pixel_num_bounding_box_area_ratio_threshold = pixel_num_bounding_box_area_ratio_threshold
        self._suspected_overlapped_cluster_size_ratio = suspected_overlapped_cluster_size_ratio

    def get_dandelion_number(
        self,
        file_path: str,
        show_picture=True,
    ) -> int:
        ROW, COLUMN = self._resize_row, self._resize_column

        original_image = cv2.imread(file_path)
        original_row, original_column, _ = original_image.shape
        resized_resolution = (ROW, COLUMN)
        if original_row > original_column:
            resized_resolution = (COLUMN, ROW)

        # resize image
        resized_original_image = cv2.resize(
            original_image,
            resized_resolution
        )
        
        # extract yellow(dandelion) pixels
        hsv_image = cv2.cvtColor(
            resized_original_image,
            cv2.COLOR_BGR2HSV
        )

        mask_image = cv2.inRange(
            hsv_image,
            self._lower_yellow,
            self._upper_yellow
        )

        target_image = cv2.bitwise_and(
            resized_original_image,
            resized_original_image,
            mask=mask_image
        )

        # noize canceling
        kernel_2x2 = np.ones((2, 2), np.uint8)

        noize_canceled_image = self._cancel_noise_with_opening(
            target_image,
            kernel_2x2,
            10
        )

        kernel_3x3 = np.ones((3, 3), np.uint8)

        noize_canceled_image = self._cancel_noise_with_opening(
            target_image,
            kernel_3x3,
            10
        )

        # get naive cluster number and cluster info list
        total_cluster_num, cluster_info_list = self._get_number_of_cluster_and_cluster_info_list(
            noize_canceled_image
        )

        # get median cluster pixel num from clusters
        sorted_cluster_info_list_by_pixel_num = sorted(
            cluster_info_list,
            key=lambda cluster_info: cluster_info.pixel_num
        )
        median_pixel_num = sorted_cluster_info_list_by_pixel_num[len(sorted_cluster_info_list_by_pixel_num)//2] \
                                .pixel_num

        # iterate cluster list to check a certain cluster is overlapped cluster
        # overlapped cluster means the cluster includes more than one dandelion
        answer = total_cluster_num
        overlapped_cluster_info_list = []
        if total_cluster_num >= 10:
            overlapped_cluster_info_list = [
                cluster_info for cluster_info in cluster_info_list
                    if self._is_overlapped_cluster(
                        cluster_info,
                        median_pixel_num
                    ) is True
            ]

        # determine each overlapped cluster has how many dandelion based on the median pixel number of clusters
        # and adjust answer
        for overlapped_cluster_info in overlapped_cluster_info_list:
            overlapped_cluster_num = max(
                round(
                    overlapped_cluster_info.pixel_num/(1 * max(median_pixel_num, 50)) # adjust when median is too small
                ),
                2
            )
            answer += (overlapped_cluster_num - 1)

            overlapped_cluster_info.dandelion_number = overlapped_cluster_num

        # just for check processed result
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
                cv2.putText(
                    resized_original_image,
                    str(cluster_info.dandelion_number),
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
            print(f'median: {median_pixel_num}')
            print(f'total cluster number: {total_cluster_num}')
            print(f'adjusted total cluster number: {answer}')

            cv2.imshow('resized_original_image_with_rect', resized_original_image)
            cv2.imshow('noize_canceled_image', noize_canceled_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return answer

    def _cancel_noise_with_opening(
        self,
        target_image: np.uint8,
        kernel: np.uint8,
        iteration_num: int
    ) -> np.uint8:
        for _ in range(iteration_num):
            noize_canceled_image = cv2.morphologyEx(
                target_image,
                cv2.MORPH_OPEN,
                kernel
            )

        return noize_canceled_image

    def _is_overlapped_cluster(
        self,
        cluster_info: ClusterInfo,
        median_pixel_num: int,
    ) -> bool:
        threshold = 100
        if cluster_info.pixel_num >= max(2*median_pixel_num, threshold):
            return True

        # judge whether the cluster is overlapped or not by two criteria
        # 1: current cluster pixel number is equal or greater than median pixel number * parameter
        # 2: pixel_num_bounding_box_area_ratio is equal or smaller than parameter
        pixel_num = cluster_info.pixel_num
        pixel_num_bounding_box_area_ratio = round(pixel_num / cluster_info.minimum_bounding_box_area * 100) / 100
        if cluster_info.pixel_num >= self._suspected_overlapped_cluster_size_ratio * median_pixel_num and \
            pixel_num_bounding_box_area_ratio <= self._pixel_num_bounding_box_area_ratio_threshold:
            
            return True

        return False

    def _get_number_of_cluster_and_cluster_info_list(
        self,
        image: np.uint8
    ) -> Tuple[int, List[dict]]:
        total_row, total_column, _ = image.shape

        visited = [[False for _ in range(total_column)] for _ in range(total_row)]
        cluster_info_list = []

        for r in range(total_row):
            for c in range(total_column):
                if np.any(image[r][c]) is np.True_ and visited[r][c] is False:
                    cluster_info = ClusterInfo()

                    self._dfs(
                        r,
                        c,
                        image,
                        visited,
                        total_row,
                        total_column,
                        cluster_info
                    )

                    # skip if cluster pixel number is too small
                    if cluster_info.pixel_num <= self._minimum_pixel_threshold:
                        continue
                    cluster_info_list.append(cluster_info)

        return len(cluster_info_list), cluster_info_list

    def _dfs(
        self,
        r: int,
        c: int,
        np_array: np.uint8,
        visited: List[List[bool]],
        total_row: int,
        total_column: int,
        cluster_info: ClusterInfo
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

            self._dfs(
                next_row,
                next_column,
                np_array,
                visited,
                total_row,
                total_column,
                cluster_info
            )


if __name__ == '__main__':
    # parameter is already optimized
    # score: 2.70
    dandelion_counter = DandelionCounter(
        lower_yellow=(20, 140, 140),
        upper_yellow=(30, 255, 255),
        minimum_pixel_threshold=8,
        pixel_num_bounding_box_area_ratio_threshold=0.7,
        suspected_overlapped_cluster_size_ratio=1.6,
    )

    OUTPUT_FILE_NAME = 'answer_without_4x4_with_threshold_with_overlap_strategy_with_area_strategy666.csv'

    all_file_path_list = [ f'./images/test40/flower ({i}).jpg' for i in range(1, 41) ]

    with open(OUTPUT_FILE_NAME, 'w', newline='') as csvfile:
        answer_csv_writer = csv.writer(csvfile, delimiter=',')

        answer_csv_writer.writerow([',', 'target'])

        for i, file_path in enumerate(all_file_path_list):
            answer = dandelion_counter.get_dandelion_number(
                file_path,
                show_picture=False
            )

            file_number = i+1
            answer_csv_writer.writerow([f'flower ({file_number}).jpg', answer])

            print(f'file number: {file_number} done')
    
    # for check
    # file_path = './images/test40/flower (24).jpg'

    # dandelion_counter = DandelionCounter(
    #     lower_yellow=(20, 140, 140),
    #     upper_yellow=(30, 255, 255),
    #     minimum_pixel_threshold=8,
    #     pixel_num_bounding_box_area_ratio_threshold=0.7,
    #     suspected_overlapped_cluster_size_ratio=1.6,
    # )

    # dandelion_counter.get_dandelion_number(
    #     file_path,
    #     show_picture=True
    # )