from warnings import catch_warnings
import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv

# folder structure
# input
# output
# fails

# ls > files.txt to have all the image names
FILES = 'input/files.txt'
THRESHOLD = 0.9

# sheet3 is the machine produced answer sheet, this has the perfect width, height
# for the region covering 4 corners
# im_orig3 = cv2.imread("sheet3.jpg")
# using that to caculate the following
WIDTH = 641
HEIGHT = 931

# coordinate intervals of corners
X_FLOOR, X_CEILING = 100, 800
Y_FLOOR, Y_CEILING = 150, 1000
# intervals of width/height of the stone corner
LOWER = 25
UPPER = 45

# number of integer in studentID
STUDENTIDnums = 6
TESTIDnums = 3
ANSWERnums = 4

ANS = ['A', 'B', 'C', 'D']
answers_091 = ['C', 'A', 'D', 'A', 'B', 'A', 'C', 'D', 'D', 'A']
answers_037 = ['D', 'B', 'C', 'B', 'B', 'C', 'D', 'B', 'D', 'D']
answers_002 = ['B', 'A', 'B', 'D', 'B', 'A', 'A', 'C', 'A', 'C']
answers_048 = ['C', 'C', 'D', 'C', 'A', 'D', 'D','D', 'B', 'D']

adict = {
    '091': answers_091,
    '037': answers_037,
    '002': answers_002,
    '048': answers_048
}

ANSSIZE = 5
ANSCOLOR = (255,0,0)
CORRECTSIZE = 3
CORRECTCOLOR = (0,255,0)

def normalize(im):
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        # plt.imshow(im_gray)
        # plt.savefig('output.png')

        # plt.show()
        blurred = cv2.GaussianBlur(im_gray, (3, 3), 0)
        return cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 77, 10)
def get_approx_contour(contour, tol=.01):
        """Gets rid of 'useless' points in the contour."""
        epsilon = tol * cv2.arcLength(contour, True)
        return cv2.approxPolyDP(contour, epsilon, True)
def get_contours(image_gray):
        contours, _ = cv2.findContours(
            image_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return map(get_approx_contour, contours)
def check_stone_coor(x,y):
    return (x <= X_FLOOR or x >= X_CEILING) and (y <= Y_FLOOR or y>= Y_CEILING)
def get_stones(im_orig, contours):
    stones = []
    for contour in contours:
        x,y,w,h = cv2.boundingRect(contour)
        if w >= LOWER and w <= UPPER and h >= LOWER and h <= UPPER and check_stone_coor(x,y):
            stones.append(np.array([x,y]))
            # print(x,y, w, h)
            # r = cv2.rectangle(im_orig, (x,y), (x+w, y+h), (255,0,0),10)
            # plt.imshow(r)
    # plt.show()
    stones = np.array(stones)
    return stones
def sort_points_counter_clockwise(points):
        origin = np.mean(points, axis=0)

        def positive_angle(p):
            x, y = p - origin
            ang = np.arctan2(y, x)
            return 2 * np.pi + ang if ang < 0 else ang

        return sorted(points, key=positive_angle)
    
# warp the distorted phone image into the perfect rectangle
def perspective_transform(img, points):
    source = np.array(
        points,
        dtype="float32")

    dest = np.array([
        [WIDTH, HEIGHT],
        [0, HEIGHT],
        [0,0],
        [WIDTH, 0]],
        dtype="float32"
    )

    transf = cv2.getPerspectiveTransform(source, dest)
    warped = cv2.warpPerspective(img, transf, (WIDTH, HEIGHT))

    return warped
# plot the bounding rectangle covering 4 corners
def plot_bounding_rect(outmost, im_orig):
    rect = cv2.minAreaRect(np.array(outmost))
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    b = cv2.drawContours(im_orig,[box],0,(0,0,255),2)
    plt.imshow(b)
    plt.show()

def get_num_strips(x1, y1, x2, y2, normalized_transf, color_transf, IDnums):
    sbdh = normalized_transf[y1:y2]
    sbd = []
    for row in sbdh:
        r = row[x1:x2]
        sbd.append(r)
    sbd = np.asarray(sbd)
    # plt.imshow(sbd)
    # plt.show()
    shei2, swid2 = sbd.shape
    # r = cv2.rectangle(color_transf, (x1,y1), (x2, y2), (255,0,0),10)
    # plt.imshow(r)
    nums_strip = []
    for i in range(IDnums):
        num_strip = []
        x1 = int(i*swid2/IDnums)
        x2 = int(swid2/IDnums + i*swid2/IDnums)
        # print(x1,x2)
        for row in sbd:
            r = row[x1:x2]
            num_strip.append(r)
        # plt.imshow(num_strip)
        # plt.show()
        nums_strip.append(num_strip)
    return nums_strip, swid2, shei2
def get_num(n_strip, shei2):
    means = []
    for i in range(10):
        num = []
        y1 = int(shei2/10 * i)
        y2 = int(shei2/10 * i + shei2/10)
        num = n_strip[y1:y2]
        mean = np.mean(num)
        means.append(mean)
    sorted_means = sorted(means)
    # print(sorted_means)
    if sorted_means[0]/sorted_means[1] >= THRESHOLD:
        return None
    return means.index(sorted_means[0])

def get_an(an_strip, awid2):
        # plt.imshow(an_strip)
        # plt.show()
        means = []
        for i in range(4):
            an = []
            x1 = int(awid2/4 * i)
            x2 = int(awid2/4 * i + awid2/4)
            for row in an_strip:
                an.append(row[x1:x2])
            # plt.imshow(an)
            # plt.show()
            mean = np.mean(an)
            means.append(mean)
        sorted_means = sorted(means)
        # print(sorted_means)
        if sorted_means[0]/sorted_means[1] >= THRESHOLD:
            return None
        return ANS[means.index(sorted_means[0])]
def get_answers_coor(swid, shei, normalized_transf):
    # cau tra loi
    ax1 = int(swid/5)
    ay1 = int(shei/1.9) -8
    ax2 = int(ax1 + swid/6) 
    ay2 = int(9.8*shei/10) + 6
    # print("cau tra loi\n")
    # print(ax1,ax2,ay1,ay2)

    # vung 2

    a2x1 = int(swid/2.3) +3
    a2x2 = int(a2x1 + swid/6) +3

    ansh = normalized_transf[ay1:ay2]
    # plt.imshow(ansh)
    # plt.show()
    return ax1, ax2, ay1, ay2, a2x1, a2x2, ansh
    
def get_ans(x1, x2, ansh):
    ans = []
    for row in ansh:
        r = row[x1:x2]
        ans.append(r)

    ans = np.asarray(ans)

    # plt.imshow(ans)
    # plt.show()

    ahei2, awid2 = ans.shape
    ans_strip = []
    for i in range(17):
        y1 = int(i*ahei2/17)
        y2 = int(ahei2/17 + i*ahei2/17)
        # print(x1,x2)
        ans_strip.append(ans[y1:y2])
        # if i == 16:
        #     plt.imshow(ans[y1:y2])
        #     plt.show()

    cans = list(map(lambda x: get_an(x, awid2), ans_strip))
    return cans, awid2, ahei2
def plot_ID(swid2, shei2, csbd, color_transf, x1, y1):
    for i, num in enumerate(csbd):
        if num != None:
            x2 = int(swid2/12 + swid2/6 * i)
            y2 = int(shei2/20 + shei2/10 * num)
            c = cv2.circle(color_transf, (x1+x2,y1+y2), 5, (255,0,0), -1)
            plt.imshow(c)
def plot_ans(ans, x1, y1, awid1, ahei1, color_transf, color = ANSCOLOR, size=ANSSIZE):
    for i, an in enumerate(ans):
        if an != None:
            y2 = int(ahei1/34 + ahei1/17 * i)
            x2 = int(awid1/8 + awid1/4 * ANS.index(an))
            c = cv2.circle(color_transf, (x1+x2,y1+y2), size, color, -1)
            plt.imshow(c)
def grading(students, answers, swid, shei, color_transf):
    points = 0
    total = 0
    for i in range(len(answers)):
        if answers[i] != None:
            # print(f'Cau {i+1}: {students[i]}, {answers[i]}')
            if students[i] == answers[i]:
                points += 1
            total += 1
    grade = round(points / total * 10,2)
    xg = int(swid/10)
    yg = int(shei/2.7)
    g = cv2.putText(color_transf, str(grade), (xg,yg), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 0 ,0), 5, cv2.LINE_AA)
    plt.imshow(g)
    return grade
def grade(inputfile):

    plt.figure(frameon=False)

    # TODO: rotate the image with cv.rotate if necessary

    im_orig = cv2.imread(f'./input/{inputfile}')


    im_normalized = normalize(im_orig)

    hei, wid = im_normalized.shape
    
    contours = get_contours(im_normalized)

    stones = get_stones(im_orig, contours)

    outmost = sort_points_counter_clockwise(stones)

    color_transf = perspective_transform(im_orig, outmost)
    normalized_transf = perspective_transform(im_normalized, outmost)

    shei, swid = normalized_transf.shape
    
    # so bao danh
    x1 = int(swid/1.7) + 12
    y1 = int(shei/29) + 2
    x2 = int(3.2*swid/4)
    y2 = int(shei/3.8) - 9
    # print("sbd\n")
    # print(x1,x2,y1,y2)
    nums_strip, swid2, shei2 = get_num_strips(x1, y1, x2, y2, normalized_transf, color_transf, STUDENTIDnums)

    csbd = list(map(lambda x: get_num(x, shei2), nums_strip))

    # ma de
    mx1 = int(4.25*swid/5) + 5
    my1 = int(shei/27.5) + 4
    mx2 = int(mx1 + swid/10)
    my2 = int(shei/3.8)-8
    # print("ma de\n")
    # print(mx1,mx2,my1,my2)
    
    mds_strip, mwid2, mhei2 = get_num_strips(mx1, my1, mx2, my2, normalized_transf, color_transf, TESTIDnums)
    # plt.imshow(mds_strip)
    # plt.show()
    # testID = input()
    
    cmd = list(map(lambda x: get_num(x, mhei2), mds_strip))
    # print('made', cmd)
    ax1, ax2, ay1, ay2, a2x1, a2x2, ansh = get_answers_coor(swid, shei, normalized_transf)
    # ar = cv2.rectangle(color_transf, (ax1, ay1), (ax2, ay2), (255,0,0),10)
    # plt.imshow(ar)
    
    ans1, awid2, ahei2 = get_ans(ax1, ax2, ansh)
    ans2, awid2, ahei2 = get_ans(a2x1, a2x2, ansh)
    # end of cau tra loi
    # print("report\n")
    # print(csbd, cmd)

    # for i in range(17):
    #     print(f'Cau {i+1}: {ans1[i]}')

    # for i in range(17):
    #     print(f'Cau {i+18}: {ans2[i]}')

    # plot the answer 
    # sbd
    x1 = int(swid/1.7) + 12
    y1 = int(shei/29) + 2
    plot_ID(swid2, shei2, csbd, color_transf, x1 ,y1)
    # plot the ma de
    mx1 = int(4.25*swid/5) + 5
    my1 = int(shei/27.5) + 4
    plot_ID(swid2, shei2, cmd, color_transf, mx1, my1)

    # plot the answers
    ax1 = int(swid/5)
    ay1 = int(shei/1.9) -8
    ax2 = int(ax1 + swid/6) 
    ay2 = int(9.8*shei/10) + 6

    ahei1 = ay2 - ay1
    awid1 = ax2 - ax1

    # vung 2

    a2x1 = int(swid/2.3) +3
    a2x2 = int(a2x1 + swid/6) +3

    
    plot_ans(ans1, ax1, ay1, awid1, ahei1, color_transf)
    plot_ans(ans2, a2x1, ay1, awid1, ahei1, color_transf)

    students = ans1 + ans2
    total = 0
    points = 0
    # print('Grading\n')
    studentID = ''.join(map(lambda x: str(x), csbd))
    if studentID == 'NoneNoneNoneNoneNoneNone':
        studentID = 'N/A'
    testID = ''.join(map(lambda x: str(x), cmd))

    answers = adict[testID]

    plot_ans(answers, ax1, ay1, awid1, ahei1, color_transf, CORRECTCOLOR, CORRECTSIZE)
    gr = grading(students, answers, swid, shei, color_transf)

    plt.axis('off')
    plt.savefig(f'./output/{inputfile}_{gr}.jpg', bbox_inches='tight', dpi=400, transparent=True, pad_inches=0)
    # plt.show() 
    plt.close()
    return [f'{inputfile}_{gr}.jpg', studentID, testID, gr]
    # return grade
