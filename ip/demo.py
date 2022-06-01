from tool.utils import *
from tool.torch_utils import *
from tool.darknet2pytorch import Darknet
import argparse
from detect_color import get_color_dist

"""hyper parameters"""
use_cuda = False


def detect_cv2(cfgfile, weightfile, imgfile):
    import cv2
    m = Darknet(cfgfile)

    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    if use_cuda:
        m.cuda()

    num_classes = m.num_classes
    if num_classes == 20:
        namesfile = 'data/voc.names'
    elif num_classes == 80:
        namesfile = '../ip/data/coco.names'
    else:
        namesfile = 'data/x.names'
    class_names = load_class_names(namesfile)

    img = cv2.imread(imgfile)
    sized = cv2.resize(img, (m.width, m.height))
    sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

    for i in range(2):
        start = time.time()
        boxes = do_detect(m, sized, 0.4, 0.6, use_cuda)
        finish = time.time()
        if i == 1:
            print('%s: Predicted in %f seconds.' % (imgfile, (finish - start)))
    # print(boxes)
    # print(boxes[0])

    _, all_list = plot_boxes_cv2(img, boxes[0], savename='./static/images/predictions.jpg', class_names=class_names)
    print(os.getcwd())
    # all list = [[sub_image of detected object, detected object name, conf_score of detected object], [sub_image of detected object2, detected object name2, conf_score of detected object2]]

    # print(all_list)
    return all_list
    # cv2.imshow("Object", all_list[0][0])
    # cv2.waitKey(0)



def get_args():
    parser = argparse.ArgumentParser('Test your image or video by trained model.')
    parser.add_argument('-cfgfile', type=str, default='./cfg/yolov4.cfg',
                        help='path of cfg file', dest='cfgfile')
    parser.add_argument('-weightfile', type=str,
                        default='./checkpoints/Yolov4_epoch1.pth',
                        help='path of trained model.', dest='weightfile')
    parser.add_argument('-input', default=None,
                        help='path of your input file.')
    parser.add_argument('-video', action="store_true", help='input as video', dest='imgfile')
    args = parser.parse_args()

    return args


def color_result(result_array, object_name=None):
    """
    This function determines the colors of object (if object is multiple,
     it will take a object which has the highest confidence score)
    Args:
        result_array:
        object_name:

    Returns: color dict of object

    """
    score = 0
    image = None

    if object_name is None:
        object_name = object_result(result_array=result_array, is_most=True)

    print(object_name, type(object_name))

    for rslt in result_array:
        if rslt[1] == object_name and score < rslt[2]:
            score = rslt[2]
            image = rslt[0]

    if image is not None:
        return get_color_dist(img=image)
    else:
        print("Object NOT FOUND !")
        return False


def object_count_result(result_array, object_name=None):
    """
    This function returns dict which has object names as keys and counts as values.
    Args:
        result_array:
        object_name:

    Returns:

    """
    count_dict = {}

    for rslt in result_array:
        if rslt[1] in count_dict:
            count_dict[rslt[1]] += 1
        else:
            count_dict[rslt[1]] = 1

    # print(object_name, type(object_name))
    # object_name = str(object_name)
    if object_name is not None and object_name in count_dict:
        return count_dict[object_name]
    elif object_name is not None and object_name not in count_dict:
        print("Object NOT FOUND !")
        return False
    else:
        return count_dict


def object_result(result_array, is_most=False):
    """
    This function returns object name which has the highest confidence score
    Args:
        result_array:
        is_most:

    Returns:

    """
    if is_most:
        obj_name = None
        score = 0

        for rslt in result_array:
            if score < rslt[2]:
                score = rslt[2]
                obj_name = rslt[1]

        if obj_name is None:
            print("Objects NOT EXISTS !")
            return False
        else:
            return obj_name

    else:
        object_list = []
        sorted_result_array = sorted(result_array, key=lambda item: item[2], reverse=True)

        for rslt in sorted_result_array:
            if rslt[1] not in object_list:
                object_list.append(rslt[1])

        if len(object_list) == 0:
            print('NOT FOUND ANY OBJECT')
            return False

        return object_list


def object_result_v2(result_array, object_count=3):
    """
    This function returns object name which has the highest confidence score
    Args:
        result_array:
        object_count: maximum number of distinct object name for return

    Returns:

    """
    if object_count < 0:
        print('KEY ERROR !')
        raise KeyError

    if object_count == 0:
        obj_name = None
        score = 0

        for rslt in result_array:
            if score < rslt[2]:
                score = rslt[2]
                obj_name = rslt[1]

        if obj_name is None:
            print("Objects NOT EXISTS !")
            return False
        else:
            return obj_name

    else:
        conf_threshold = 0.4
        object_list = []
        sorted_result_array = sorted(result_array, key=lambda item: item[2], reverse=True)
        # print(sorted_result_array)
        count = 0

        for rslt in sorted_result_array:
            if rslt[2] > conf_threshold:
                if rslt[1] not in object_list:
                    object_list.append(rslt[1])
                    count = len(object_list)

            if count >= object_count:
                break

        if len(object_list) == 0:
            print('NOT FOUND ANY OBJECT')
            return False

        return object_list


if __name__ == '__main__':
    args = get_args()
    all_lst = detect_cv2(args.cfgfile, args.weightfile, args.input)
    print(color_result(result_array=all_lst, object_name='cup'))
    print(object_result(result_array=all_lst))
    print(object_count_result(result_array=all_lst))

