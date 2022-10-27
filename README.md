# omr
scan answer sheet.jpg and grade it

# logic
using opencv to process the images

# flow
- find contours, plot the bounding rectangle, look for 4 black stones at the corners
- since images can be taken with a phone and be distorted a bit: use a transformation function that take in 4 top-left corners of the 4 stones and warp them back into the should-be rectangle  
- from there, calculate the studentID area, the testID area, and the answer areas
- for the studentID area: cut the area into strips of numbers, cut them into each number, take the pixel values array and average them. chosen number has lower pixel value than not marked ones (i wonder why it is lower, not higher, since it is marked black, it should have higher pixel value, ??!)
- same for other areas: calculate the coordinates to cut them into strips, then in each strip, cut them smaller into each answer, then average the pixel value. the value that stands out is the marked one.
- marking more than one answer is counted as marking none

