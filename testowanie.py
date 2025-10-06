
import cv2
import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib as mpl
from PIL import Image
import matplotlib.pyplot as plt
from utils.func import (get_pictures, apply_hough_transformation, group_lines,
                        transform_annotation)
from utils.lines import Line
from utils.corners import CourtFinder

mpl.rcParams['image.cmap'] = 'gray'

with open('test.config.json') as file:
    config = json.load(file)

train_pics = get_pictures('pics/compliant')['rgb']
test_df_rows = []

for i, (data, train_pic) in enumerate(zip(config['data'], train_pics)):
    path = Path(f"{config['testing_pics_dir']}/{config['commit']}")
    path.mkdir(exist_ok = True, parents = True)

    
    train_pic_hough_lines, line_endpoints = apply_hough_transformation(train_pic)
    line_objs = [Line.from_hough_line(line[0]) for line in line_endpoints]
    line_objs = [line for line in line_objs if line.xv is None]

    grouped_lines = group_lines(line_objs)


    train_pic_line_intersections = train_pic.copy()
    intersections = []
    for group1 in grouped_lines:
        for group2 in grouped_lines:
            intersection = group1.intersection(group2, train_pic_line_intersections)
            if intersection is not None:
                intersections.append(intersection)


    candidates = CourtFinder(intersections, train_pic)

    try:
        # closer_outer_baseline_point, closer_outer_netline_point, closer_doubles_sideline -------
        closer_outer_baseline_intersecetion, closer_outer_netintersection, used_line = candidates.find_closer_outer_baseline_point()
        closer_outer_baseline_point = closer_outer_baseline_intersecetion.point
        closer_outer_netline_point = closer_outer_netintersection.point
        closer_doubles_sideline = Line.from_points(closer_outer_baseline_point, closer_outer_netline_point)

        gt_closer_outer_baseline_point = transform_annotation(train_pic, data['closer_outer_baseline_point'])
        gt_closer_outer_netline_point = transform_annotation(train_pic, data['closer_outer_netline_point'])

        closer_outer_baseline_point_dist = gt_closer_outer_baseline_point.distance(closer_outer_baseline_point)
        closer_outer_netline_point_dist = gt_closer_outer_netline_point.distance(closer_outer_netline_point)

        # further_outer_baseline_point, baseline  -------
        further_outer_baseline_intersection, last_local_line = candidates.find_further_outer_baseline_intersection(closer_outer_baseline_intersecetion, used_line, data['canny_thresh']['lower'] , data['canny_thresh']['upper'], offset=data['offset'])
        further_outer_baseline_point = further_outer_baseline_intersection.point
        baseline = Line.from_points(closer_outer_baseline_point, further_outer_baseline_point)

        gt_further_outer_baseline_point = transform_annotation(train_pic, data['further_outer_baseline_point'])
        further_outer_baseline_point_dist = gt_further_outer_baseline_point.distance(further_outer_baseline_point)

        # netline -----
        netline = candidates.find_netline(closer_outer_netline_point, baseline, data['max_line_gap'])


        # further_doubles_sideline, further_outer_netline_point ---
        further_doubles_sideline = candidates.find_further_doubles_sideline(further_outer_baseline_point, last_local_line, data['offset'], data['extra_offset'], data['bin_thresh'], data['surface_type'])
        further_outer_netline_point = further_doubles_sideline.intersection(netline, train_pic).point

        gt_further_outer_netline_point = transform_annotation(train_pic, data['further_outer_netline_point'])
        further_outer_netline_point_dist = gt_further_outer_netline_point.distance(further_outer_netline_point)


        # inner points
        closer_inner_baseline_point, further_inner_baseline_point = candidates.scan_endline(baseline,
                                                                                            netline, 
                                                                                            closer_outer_baseline_point, 
                                                                                            further_outer_netline_point, 
                                                                                            closer_outer_netline_point, 
                                                                                            further_outer_baseline_point, 
                                                                                            data['bin_thresh_endline_scan']['baseline'],
                                                                                            data['canny_thresh']['lower'],
                                                                                            data['canny_thresh']['upper'],
                                                                                            data['max_line_gap'],
                                                                                            searching_line = 'base',
                                                                                            )


        closer_inner_netline_point, further_inner_netline_point = candidates.scan_endline(baseline, 
                                                                                netline, 
                                                                                closer_outer_baseline_point, 
                                                                                further_outer_netline_point, 
                                                                                closer_outer_netline_point, 
                                                                                further_outer_baseline_point, 
                                                                                data['bin_thresh_endline_scan']['netline'],
                                                                                data['canny_thresh']['lower'],
                                                                                data['canny_thresh']['upper'],
                                                                                data['max_line_gap'],
                                                                                searching_line = 'net',
                                                                                )
        
        # net_service_point and centre_service_line
        net_service_point, centre_service_line = candidates.find_net_service_point_centre_service_line(closer_outer_baseline_point, 
                                                                                closer_outer_netline_point, 
                                                                                further_outer_baseline_point,
                                                                                further_outer_netline_point, 
                                                                                closer_inner_baseline_point,
                                                                                further_inner_baseline_point,
                                                                                closer_inner_netline_point,
                                                                                further_inner_netline_point,
                                                                                baseline, 
                                                                                netline,
                                                                                data['bin_thresh_centre_service_line'],
                                                                                data['canny_thresh']['lower'],
                                                                                data['canny_thresh']['upper'],
                                                                                data['max_line_gap_centre_service_line'],
                                                                                data['min_line_len_ratio'],
                                                                                data['hough_thresh'],
                                                                                )
        

        gt_closer_inner_baseline_point = transform_annotation(train_pic, data['closer_inner_baseline_point'])
        closer_inner_baseline_point_dist = gt_closer_inner_baseline_point.distance(closer_inner_baseline_point)

        gt_further_inner_baseline_point = transform_annotation(train_pic, data['further_inner_baseline_point'])
        further_inner_baseline_point_dist = gt_further_inner_baseline_point.distance(further_inner_baseline_point)

        gt_closer_inner_netline_point = transform_annotation(train_pic, data['closer_inner_netline_point'])
        closer_inner_netline_point_dist = gt_closer_inner_netline_point.distance(closer_inner_netline_point)

        gt_further_inner_netline_point = transform_annotation(train_pic, data['further_inner_netline_point'])
        further_inner_netline_point_dist = gt_further_inner_netline_point.distance(further_inner_netline_point)

        closer_singles_sideline = Line.from_points(closer_inner_baseline_point, closer_inner_netline_point)
        further_singles_sideline = Line.from_points(further_inner_baseline_point, further_inner_netline_point)

        pic = train_pic.copy()
        pts1 = closer_doubles_sideline.limit_to_img(pic)
        pts2 = baseline.limit_to_img(pic)
        pts3 = netline.limit_to_img(pic)
        pts4 = further_doubles_sideline.limit_to_img(pic)
        pts5 = closer_singles_sideline.limit_to_img(pic)
        pts6 = further_singles_sideline.limit_to_img(pic)
        pts7 = centre_service_line.limit_to_img(pic)

        cv2.line(pic, *pts1, (0, 0, 255))
        cv2.line(pic, *pts2, (0, 0, 255))
        cv2.line(pic, *pts3, (0, 0, 255))
        cv2.line(pic, *pts4, (0, 0, 255))
        cv2.line(pic, *pts5, (0, 0, 255))
        cv2.line(pic, *pts6, (0, 0, 255))
        cv2.line(pic, *pts7, (0, 0, 255))
        
        cv2.circle(pic, closer_outer_netline_point, 1, (255, 0,0), 3, -1)
        cv2.circle(pic, closer_outer_baseline_point, 1, (0,255,0), 3, -1)
        cv2.circle(pic, further_outer_baseline_point, 1, (0,0,255), 3, -1)
        cv2.circle(pic, further_outer_netline_point, 1, (255,0,0), 3, -1)

        cv2.circle(pic, closer_inner_baseline_point, 1, (255,0,0), 3, -1)
        cv2.circle(pic, closer_inner_netline_point, 1, (255,0,0), 3, -1)
        cv2.circle(pic, further_inner_baseline_point, 1, (255,0,0), 3, -1)
        cv2.circle(pic, further_inner_netline_point, 1, (255,0,0), 3, -1)

        cv2.circle(pic, net_service_point, 1, (255,0,0), 3, -1)

        # for inter in intersections:
        #     cv2.circle(pic, inter.point, 1, (0, 0, 0))

        Image.fromarray(pic).save(path / data['pic_name'])

    except Exception as e:
        print(e)
        closer_outer_baseline_point_dist = None
        closer_outer_netline_point_dist = None

    finally:
        test_df_rows.append([i, data['pic_name'], data['bin_thresh'], 
                             closer_outer_baseline_point_dist, 
                             closer_outer_netline_point_dist, 
                             further_outer_baseline_point_dist, 
                             further_outer_netline_point_dist,
                             closer_inner_baseline_point_dist,
                             further_inner_baseline_point_dist,
                             closer_inner_netline_point_dist,
                             further_inner_netline_point_dist
                             ])

pd.DataFrame(test_df_rows, columns=['pic_index', 'pic_name', 'bin_thresh', 
                                    'closer_outer_baseline_point_dist', 
                                    'closer_outer_netline_point_dist', 
                                    'further_outer_baseline_point_dist', 
                                    'further_outer_netline_point_dist',
                                    'closer_inner_baseline_point',
                                    'further_inner_baseline_point',
                                    'closer_inner_netline_point',
                                    'further_inner_netline_point',
                                    ]).to_csv(path / "test_df.csv", index=False)
