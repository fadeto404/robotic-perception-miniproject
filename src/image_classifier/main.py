"""
MADE AND SUBMITTED BY: Rasmus Nymark Skjelsager

Image processing pipeline for finding a few different kinds of traffic signs.
Is made for finding:
    * Speed limit signs         :   A circular sign with a red border
    * Driving direction sign    :   A circular sign which is predominantly blue, usually with a white arrow
    * Distance/direction sign   :   A rectangular sign that leads to points of interest (only blue ones chosen for simplicity)
    * Right-of-way sign         :   A triangular sign, pointing downwards, red border and completely white inside
A few other types of signs can be found in the dataset as well, but are not meant to be recognised by the classifier.

There is a manually made classification process which writes in RED on the image.

There is also a self-implemented nearest-neighbour classifier which writes in GREEN on the image.
The nearest-neighbour classifier uses a model of each sign which is simply an arbitrarily chosen copy of a feature
vector of such an object from the function get_object_features().

Be aware: requires patience - it's not fast

============ HOW TO RUN ============
1. At the bottom of the file, find the function main()
2. In the function, find the variable images, with the function call get_image_paths()
3. Pass the path to your data directory to the function (datatype = string)
4. Run the file (made for python 3.7)
"""

import cv2

from os import walk, path

import numpy as np

from matplotlib import pyplot as plt


class ImageProcessingPipeline(object):
    # Constructor
    def __init__(self):
        # Values for keeping track of recognised objects
        self.speed_limit = 0
        self.driving_dir = 0
        self.dist_dir = 0
        self.right_of_way = 0
        self.unknown_objects = 0

    def pre_process_image(self, img, gamma=1.0):
        # Contrast/brightness correction, reduce contrast a little, increase brightness a little
        # Values found through testing
        new_img = cv2.convertScaleAbs(img, alpha=0.8, beta=25)

        # Gamma correction, unused
        look_up_table = np.empty((1, 256), np.uint8)
        for i in range(256):
            look_up_table[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
        res = cv2.LUT(new_img, look_up_table)
        # self.show(res)
        return new_img

    # This method binarizes the image and applies both a red and a blue threshold
    # Both binary masks are returned separately
    def get_binary_masks(self, img):
        blue_lower_threshold = (70, 0, 0)   # BGR-format
        blue_upper_threshold = (255, 95, 30)

        red_lower_threshold = (0, 0, 104)
        red_upper_threshold = (99, 81, 255)
        red_mask = cv2.inRange(img, red_lower_threshold, red_upper_threshold)

        blue_mask = cv2.inRange(img, blue_lower_threshold, blue_upper_threshold)
        # compound_mask = cv2.bitwise_or(red_mask, blue_mask)

        return blue_mask, red_mask

    # Method for filtering out noise from the binarized images
    def filter_noise(self, binary_image):
        # Median filter for removing s&p noise
        noise_filtered = cv2.medianBlur(binary_image, 11)

        # Morphology for increasing quality of remaining binary objects
        circ_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        opened = cv2.morphologyEx(noise_filtered, cv2.MORPH_OPEN, circ_kernel)  # Opening (erosion + dilation)

        big_circ_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (150, 150))  # Oof, needs performance
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, big_circ_kernel)  # Closing (dilation + erosion)
        # self.show(closed)
        return closed

    # Finds the blobs in the image
    def get_blobs(self, binary_image):
        # Find edges of binary image using the Laplacian algorithm
        # Unused
        # edges = cv2.Laplacian(binary_image, 8)

        # Gets the contours of the blobs, and a hierarchy. Hierarchy is not relevant when using cv2.RETR_EXTERNAL
        contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnt = []

        # Filter out very small objects (area of 1000 px or less)
        for contour in contours:
            if cv2.contourArea(contour) > 1000:
                cnt.append(contour)
        return cnt

    # Method for calculating various features of blobs
    def get_object_features(self, contours, original_image):
        objects_with_features = []
        for contour in contours:
            feature_vector = []
            # Compute image moments
            moments = cv2.moments(contour)
            area = moments['m00']

            # Get length of blob edge
            perimeter = cv2.arcLength(contour, True)

            circularity = 4*np.pi*area/(perimeter**2)

            # Get bounding box
            box_x, box_y, w, h = cv2.boundingRect(contour)

            # Width/height ratio
            aspect_ratio = w/h

            compactness = area/(w*h)

            # Bounding circle
            (circ_x, circ_y), radius = cv2.minEnclosingCircle(contour)
            radius = int(radius)
            circle_area_overlap = area/(np.pi*radius**2)

            # Bounding triangle
            triangle_area, triangle = cv2.minEnclosingTriangle(contour)
            triangle_area_overlap = area/triangle_area

            # Finds the average of each colour channel inside the contour
            mask = np.zeros((original_image.shape[0], original_image.shape[1]), np.uint8)
            cv2.drawContours(mask, [contour], 0, 255, -1)
            mean_of_colours = cv2.mean(original_image, mask=mask)

            # Distance from center of mass to center of bounding box
            # Scaled to account for differing box areas
            com = [int(moments['m10'] / moments['m00']), int(moments['m01'] / moments['m00'])]
            box_center_distance = np.sqrt((com[0]-box_x)**2 + (com[1]-box_y)**2)/(w*h)

            # Distance from center of mass to center of bounding circle
            circle_center_distance = np.sqrt((com[0]-int(circ_x))**2 + (com[1]-int(circ_y))**2)/(np.pi*radius**2)

            feature_vector = [circularity,
                              aspect_ratio,
                              circle_area_overlap,
                              triangle_area_overlap,
                              mean_of_colours,
                              compactness,
                              box_center_distance,
                              circle_center_distance]
            objects_with_features.append(feature_vector)

        return objects_with_features

    def classify(self, image, blobs, feature_vectors):
        # Iterates through every object
        for i in range(0, len(blobs)):
            vector = feature_vectors[i]     # Gets the feature vector in question
            blob = blobs[i]                 # Gets the corresponding object
            # Gets and draws the bounding box of the object on the original image
            x, y, w, h = cv2.boundingRect(blob)
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 5)

            # Simple, manually made classification process
            # A more advanced classifier, could be made
            # Various checks that indicate this is a circle which is more red
            if (vector[0] >= 0.8) and (0.5 <= vector[1] <= 1.5) and (vector[5] >= 0.6) and (vector[4][0] < vector[4][2] > 100):
                # Puts text on the image corresponding to the class
                cv2.putText(image, 'Speed limit sign', (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                print("Speed limit sign")
                self.speed_limit += 1

            # Various checks that indicate this is a circle which is more blue
            elif (vector[0] >= 0.8) and (0.5 <= vector[1] <= 1.5) and (vector[2] >= 0.6) and (vector[4][0] > vector[4][2]):
                cv2.putText(image, 'Driving direction sign', (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                print("Driving direction sign")
                self.driving_dir += 1

            # Various checks that indicate this is a rectangle which is more blue
            elif (vector[0] < 0.5) and (vector[1] >= 4) and (vector[4][0] > vector[4][2]) and (vector[5] >= 0.6):
                cv2.putText(image, 'Direction/distance sign', (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                print("Direction/distance sign")
                self.dist_dir += 1

            # Various checks that indicate this is a triangle which is more red
            elif (vector[1] >= 0.5) and (vector[3] >= 0.9) and (vector[4][0] < vector[4][2]):
                cv2.putText(image, 'Right-of-way sign', (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                print("Right of way sign")
                self.right_of_way += 1

            # If not belonging to these classes, set unknown
            else:
                cv2.putText(image, 'Unknown object', (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                print("Unknown object")
                self.unknown_objects += 1
            print(vector)

        # Show image with contours and written class for each object
        self.show(image)
        return

    # Quick and dirty self-made implementation of a nearest neighbour classifier
    # Does not work with "unknown" objects. Everything is a traffic sign!
    def model_classifier(self, image, blobs, feature_vectors):
        # Models are not trained, but simply copied output from the get_object_features() function to illustrate the
        # concept.
        models = [[0.28551925482222557, 8.402877697841726, 0.13849560216934478, 0.5296973600434626,
                               123.97984566210351 / 100, 110.34765518655324 / 100, 95.9113363739333 / 100, 0.0,
                               0.9124518022568247, 0.00180073318356072, 1.3628494201740128e-05],
                   [0.9090101386228023, 0.9583333333333334, 0.9670873278153362, 0.6079159339108535,
                               108.7198318149965 / 100, 71.56748423265591 / 100, 44.746881569726696 / 100, 0.0,
                               0.792572463768116, 0.007286073615577975, 0.0],
                  [0.5913786773655558, 1.0106382978723405, 0.4708543640433772, 0.9769805184335513,
                            105.89096177238372 / 100, 98.01868328618093 / 100, 143.07054751458992 / 100, 0.0,
                            0.5102358622620381, 0.0008591321414850825, 5.976466240828449e-05],
                  [0.8902884690209287, 1.007070707070707, 0.8691871284942713, 0.6072522408984792,
                            121.04048987622961 / 100, 110.07341116210917 / 100, 146.57625914353645 / 100, 0.0,
                            0.7771136642249983, 0.0007106730589626636, 8.012783162665594e-06]
                  ]
        # Classify every vector
        for i in range(0, len(blobs)):
            vector = feature_vectors[i]     # Gets the feature vector in question
            blob = blobs[i]                 # Gets the corresponding object
            # Gets and draws the bounding box of the object on the original image
            x, y, w, h = cv2.boundingRect(blob)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 5)

            unpacked_vector = [vector[0],
                               vector[1],
                               vector[2],
                               vector[3],
                               vector[4][0] / 100,
                               vector[4][1] / 100,
                               vector[4][2] / 100,
                               vector[4][3],
                               vector[5],
                               vector[6],
                               vector[7]
                               ]
            # Compute distance to every object in the model (Euclidean distance)
            dists = []
            for k in range(0, 4):
                sums = 0
                for j in range(0, len(unpacked_vector)):
                    sums += (unpacked_vector[j] - models[k][j])**2
                dists.append(np.sqrt(sums))

            min = dists[0]
            min_index = 0
            for k in range(1, 4):
                if min > dists[k]:
                    min = dists[k]
                    min_index = k
            # Writes the result in green text on the image
            if min_index == 0:
                cv2.putText(image, 'Direction/distance sign', (x + 30, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                            cv2.LINE_AA)
                print("Modelled as Direction/distance sign")
            elif min_index == 1:
                cv2.putText(image, 'Driving direction sign', (x + 30, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                            cv2.LINE_AA)
                print("Modelled as Driving direction sign")
            elif min_index == 2:
                cv2.putText(image, 'Right-of-way sign', (x + 30, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                            cv2.LINE_AA)
                print("Modelled as Right-of-way sign")
            elif min_index == 3:
                cv2.putText(image, 'Speed limit sign', (x + 30, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                            cv2.LINE_AA)
                print("Modelled as Speed limit sign")

        # Show image with contours and written class for each object
        self.show(image)
        return


    # Method called to start the process
    def process_image(self, image_path):
        image = cv2.imread(image_path)
        image = self.pre_process_image(image)
        blue_masked_bin_image, red_masked_bin_image = self.get_binary_masks(image)

        # Gets objects and features from "blue-thresholded" binary mask
        blue_filtered = self.filter_noise(blue_masked_bin_image)
        blue_blobs = self.get_blobs(blue_filtered)
        blue_feature_vectors = self.get_object_features(blue_blobs, image)

        # Gets objects and features from "red-thresholded" binary mask
        red_filtered = self.filter_noise(red_masked_bin_image)
        red_blobs = self.get_blobs(red_filtered)
        red_feature_vectors = self.get_object_features(red_blobs, image)

        # Concatenates the two datasets and starts classifying
        blobs = blue_blobs + red_blobs
        feature_vectors = blue_feature_vectors + red_feature_vectors
        self.classify(image, blobs, feature_vectors)
        self.model_classifier(image, blobs, feature_vectors)

    # Method for displaying images easily
    def show(self, img, name='Result'):
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(name, 1500, 1020)
        cv2.imshow(name, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    # Returns the total amount of blobs found, the amount that was put in a class, the amount that was not classified
    # and a number of each sign
    def get_stats(self):
        return {"Total objects found in dataset" : self.speed_limit + self.right_of_way + self.dist_dir + self.driving_dir + self.unknown_objects,
                "Classified objects" : self.speed_limit + self.right_of_way + self.dist_dir + self.driving_dir,
                "Unclassified objects" : self.unknown_objects,
                "Speed limit signs" : self.speed_limit,
                "Driving direction signs" : self.driving_dir,
                "Direction/distance signs" : self.dist_dir,
                "Right-of-way signs" : self.right_of_way
                }


# Helper function for creating a histogram of greyscale values
def gray_histogram(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    color = ('i')
    for i, col in enumerate(color):
        histr = cv2.calcHist([img], [i], None, [256], [0, 256])
        plt.plot(histr, color='k')
        plt.xlim([0, 256])
    plt.show()
    return histr


# Helper function for creating a histogram of BGR values
def bgr_histogram(image):
    img = image
    color = ('b', 'g', 'r')
    for i, col in enumerate(color):
        histr = cv2.calcHist([img], [i], None, [256], [0, 256])
        plt.plot(histr, color=col)
        plt.xlim([0, 256])
    plt.show()
    return histr


# Helper function for returning paths to images in a directory. Uses all files, so don't put non-images in the folder
def get_image_paths(data_path):
    image_paths = []
    for root, directories, filenames in walk(data_path):
        for file in filenames:
            image_paths.append(path.join(root, file))

    return image_paths


# Main function. Run this to classify. Change path in get_image_paths() to your path
def main():
    classifier = ImageProcessingPipeline()
    images = get_image_paths('/home/rns/Documents/Robotics/semester4/Robotic Perception/miniproject/data/')
    for img in images:
        classifier.process_image(img)

    print("Result of manually implemented classifier: ")
    print(classifier.get_stats())



if __name__ == '__main__':
    main()

"""
Resources:
https://docs.opencv.org/4.2.0/d5/daf/tutorial_py_histogram_equalization.html
https://docs.opencv.org/4.2.0/d2/d96/tutorial_py_table_of_contents_imgproc.html
"""
