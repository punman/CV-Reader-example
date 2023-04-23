import cv2                      # OpenCV bindings
import numpy as np
from collections import Counter


class BackgroundColorDetector():
    def __init__(self, image):
        self.img = image
        self.manual_count = {}
        self.w, self.h = self.img.shape
        self.total_pixels = self.w*self.h

    def count(self):
        for y in range(0, self.h):
            for x in range(0, self.w):
                RGB = (self.img[x, y], self.img[x, y], self.img[x, y])
                if RGB in self.manual_count:
                    self.manual_count[RGB] += 1
                else:
                    self.manual_count[RGB] = 1

    def average_colour(self):
        red = 0
        green = 0
        blue = 0
        sample = 10
        for top in range(0, sample):
            red += self.number_counter[top][0][0]
            green += self.number_counter[top][0][1]
            blue += self.number_counter[top][0][2]

        average_red = red / sample
        average_green = green / sample
        average_blue = blue / sample
        print("Average RGB for top ten is: (", average_red,
              ", ", average_green, ", ", average_blue, ")")
        return (average_red, average_green, average_blue)

    def twenty_most_common(self):
        self.count()
        self.number_counter = Counter(self.manual_count).most_common(20)
        # for rgb, value in self.number_counter:
        #     print(rgb, value, ((float(value)/self.total_pixels)*100))

    def detect(self):
        self.twenty_most_common()
        self.percentage_of_first = (
            float(self.number_counter[0][1])/self.total_pixels)
        # print(self.percentage_of_first)
        if self.percentage_of_first > 0.5:
            # print("Background color is ", self.number_counter[0][0])
            return self.number_counter[0][0]
        else:
            return self.average_colour()
            


if __name__ == "__main__":
    
    img = "output/threshold/output_threshold6.jpg"
    BackgroundColor = BackgroundColorDetector(img)
    BackgroundColor.detect()