import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.figure(frameon=False)

im_orig = cv2.imread("sheet8.jpg")
# im_orig = cv2.imread("test.jpeg")
im_orig3 = cv2.imread("sheet3.jpg")

def normalize(im):
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(im_gray, (3, 3), 0)
    return cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 77, 10)

im_normalized = normalize(im_orig)
im_normalized3 = normalize(im_orig3)

hei, wid = im_normalized.shape

def get_approx_contour(contour, tol=.01):
    """Gets rid of 'useless' points in the contour."""
    epsilon = tol * cv2.arcLength(contour, True)
    return cv2.approxPolyDP(contour, epsilon, True)
def get_contours(image_gray):
    contours, _ = cv2.findContours(
        image_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return map(get_approx_contour, contours)

contours = get_contours(im_normalized)
contours3 = get_contours(im_normalized3)

# contours = list(contours)
stones = []
for contour in contours:
    # area = cv2.contourArea(corner_contour)
    x,y,w,h = cv2.boundingRect(contour)
    # if w >= wid/5 and w<=wid/2 and h >= hei/5 and h<= hei/2:
        # print(x,y,w,h)
        # r = cv2.rectangle(im_orig, (x,y), (x+w, y+h), (255,0,0),10)
    if w >= 30 and w <= 50 and h >= 30 and h<=50:
    # if w>=800 and h>=1000:
        stones.append(np.array([x,y]))
        # print(x,y, w, h)
        # r = cv2.rectangle(im_orig, (x,y), (x+w, y+h), (255,0,0),10)
        # plt.imshow(r)
# plt.show()

stones = np.array(stones)

stones3 = []
for contour in contours3:
    # area = cv2.contourArea(corner_contour)
    x,y,w,h = cv2.boundingRect(contour)
    # if w >= wid/5 and w<=wid/2 and h >= hei/5 and h<= hei/2:
        # print(x,y,w,h)
        # r = cv2.rectangle(im_orig, (x,y), (x+w, y+h), (255,0,0),10)
    if w >= 25 and w <= 50 and h >= 25 and h<=50:
    # if w>=800 and h>=1000:
        stones3.append(np.array([x,y]))
        # print(x,y, w, h)
        # r = cv2.rectangle(im_orig3, (x,y), (x+w, y+h), (255,0,0),10)
        # plt.imshow(r)
# plt.show()

stones3 = np.array(stones3)

# def calculate_contour_features(contour):
#     moments = cv2.moments(contour)
#     return cv2.HuMoments(moments)


# def calculate_corner_features():
#     corner_img = cv2.imread('img/corner.png')
#     corner_img_gray = cv2.cvtColor(corner_img, cv2.COLOR_BGR2GRAY)
#     contours, hierarchy = cv2.findContours(
#         corner_img_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#     if len(contours) != 2:
#         raise RuntimeError(
#             'Did not find the expected contours when looking for the corner')

#     corner_contour = next(ct
#                           for i, ct in enumerate(contours)
#                           if hierarchy[0][i][3] != -1)

#     return calculate_contour_features(corner_contour)
# def features_distance(f1, f2):
#     return np.linalg.norm(np.array(f1) - np.array(f2))

# def get_corners(contours):

#     corner_features = calculate_corner_features()
#     return sorted(
#         contours,
#         key=lambda c: features_distance(
#                 corner_features,
#                 calculate_contour_features(c)))[:4]
# corners = get_corners(contours)

# cv2.drawContours(im_orig, corners, -1, (0, 255, 0), 3)
def sort_points_counter_clockwise(points):
    origin = np.mean(points, axis=0)

    def positive_angle(p):
        x, y = p - origin
        ang = np.arctan2(y, x)
        return 2 * np.pi + ang if ang < 0 else ang

    return sorted(points, key=positive_angle)
# def get_bounding_rect(contour):
#     rect = cv2.minAreaRect(contour)
#     box = cv2.boxPoints(rect)
#     return np.int0(box)

# def get_outmost_points(contours):
#     all_points = np.concatenate(contours)
#     return get_bounding_rect(all_points)

# outmost = sort_points_counter_clockwise(get_outmost_points(corners))

outmost = sort_points_counter_clockwise(stones)

outmost3 = sort_points_counter_clockwise(stones3)

# outmost = sorted(dict(stones).items())
# outmost = [outmost[0], [outmost[1][0], outmost[1][1]+33], [outmost[2][0] + 33, outmost[2][1]+33], [outmost[3][0]+33, outmost[3][1]]]
rect = cv2.minAreaRect(np.array(outmost))
box = cv2.boxPoints(rect)
box = np.int0(box)
# b = cv2.drawContours(im_orig,[box],0,(0,0,255),2)
# boundary = cv2.rectangle(corner_img, (x,y), (x+w, y+h), (255,0,0),10)
# plt.imshow(b)
# plt.show()

rect3 = cv2.minAreaRect(np.array(outmost3))
box3 = cv2.boxPoints(rect3)
box3 = np.int0(box3)
# b3 = cv2.drawContours(im_orig,[box3],0,(0,0,255),2)
# boundary = cv2.rectangle(corner_img, (x,y), (x+w, y+h), (255,0,0),10)

# plt.imshow(b3)
# plt.show()
TRANSF_SIZE = 512

WIDTH = 641
HEIGHT = 931

def perspective_transform(img, points):
    source = np.array(
        points,
        dtype="float32")

    dest = np.array([
        [TRANSF_SIZE, TRANSF_SIZE],
        [0, TRANSF_SIZE],
        [0, 0],
        [TRANSF_SIZE, 0]],
        dtype="float32")

    dest2 = np.array([
        [WIDTH, HEIGHT],
        [0, HEIGHT],
        [0,0],
        [WIDTH, 0]],
        dtype="float32"
    )

    # transf = cv2.getPerspectiveTransform(source, dest)
    # warped = cv2.warpPerspective(img, transf, (TRANSF_SIZE, TRANSF_SIZE))
    transf = cv2.getPerspectiveTransform(source, dest2)
    warped = cv2.warpPerspective(img, transf, (WIDTH, HEIGHT))

    return warped
color_transf = perspective_transform(im_orig, outmost)
normalized_transf = perspective_transform(im_normalized, outmost)


shei, swid = normalized_transf.shape
# so bao danh

x1 = int(swid/1.7) + 12
y1 = int(shei/29) + 2
x2 = int(3.2*swid/4)
y2 = int(shei/3.8) - 9
print("sbd\n")
print(x1,x2,y1,y2)

sbdh = normalized_transf[y1:y2]
sbd = []
for row in sbdh:
    r = row[x1:x2]
    sbd.append(r)

sbd = np.asarray(sbd)

plt.imshow(sbd)
plt.show()

shei2, swid2 = sbd.shape
nums_strip = []
for i in range(6):
    num_strip = []
    x1 = int(i*swid2/6)
    x2 = int(swid2/6 + i*swid2/6)
    # print(x1,x2)
    for row in sbd:
        r = row[x1:x2]
        num_strip.append(r)
    plt.imshow(num_strip)
    # plt.show()
    nums_strip.append(num_strip)

def get_num(n_strip):
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
    if sorted_means[0]/sorted_means[1] >= 0.85:
        return None
    return means.index(sorted_means[0])

csbd = list(map(get_num, nums_strip))

# end of sbd

# ma de

mx1 = int(4.25*swid/5) + 5
my1 = int(shei/27.5) + 4
mx2 = int(mx1 + swid/10)
my2 = int(shei/3.8)-8
print("ma de\n")
print(mx1,mx2,my1,my2)

mdh = normalized_transf[my1:my2]
md = []
for row in mdh:
    r = row[mx1:mx2]
    md.append(r)

md = np.asarray(md)

plt.imshow(md)
plt.show()

mhei2, mwid2 = md.shape
mds_strip = []
for i in range(3):
    md_strip = []
    x1 = int(i*mwid2/3)
    x2 = int(mwid2/3 + i*mwid2/3)
    # print(x1,x2)
    for row in md:
        r = row[x1:x2]
        md_strip.append(r)
    plt.imshow(md_strip)
    # plt.show()
    mds_strip.append(md_strip)

def get_md(md_strip):
    means = []
    for i in range(10):
        num = []
        y1 = int(mhei2/10 * i)
        y2 = int(mhei2/10 * i + mhei2/10)
        num = md_strip[y1:y2]
        mean = np.mean(num)
        means.append(mean)
    sorted_means = sorted(means)
    if sorted_means[0]/sorted_means[1] >= 0.87:
        return None
    return means.index(sorted_means[0])

cmd = list(map(get_md, mds_strip))
# end of ma de

# cau tra loi

ax1 = int(swid/5)
ay1 = int(shei/1.9) -8
ax2 = int(ax1 + swid/6) 
ay2 = int(9.8*shei/10) + 6
print("cau tra loi\n")
print(ax1,ax2,ay1,ay2)

# vung 2

a2x1 = int(swid/2.3) +3
a2x2 = int(a2x1 + swid/6) +3

ansh = normalized_transf[ay1:ay2]
plt.imshow(ansh)
# plt.show()
ANS = ['A', 'B', 'C', 'D']
def get_ans(x1, x2):
    ans = []
    for row in ansh:
        r = row[x1:x2]
        ans.append(r)

    ans = np.asarray(ans)

    plt.imshow(ans)
    plt.show()

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

    def get_an(an_strip):
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
        if sorted_means[0]/sorted_means[1] >= 0.87:
            return None
        return ANS[means.index(sorted_means[0])]

    cans = list(map(get_an, ans_strip))
    return cans
ans1 = get_ans(ax1, ax2)
ans2 = get_ans(a2x1, a2x2)
# end of cau tra loi
print("report\n")
print(csbd, cmd)

for i in range(17):
    print(f'Cau {i+1}: {ans1[i]}')

for i in range(17):
    print(f'Cau {i+18}: {ans2[i]}')

# plot the answer 
# sbd

x1 = int(swid/1.7) + 12
y1 = int(shei/29) + 2

for i, num in enumerate(csbd):
    if num != None:
        x2 = int(swid2/12 + swid2/6 * i)
        y2 = int(shei2/20 + shei2/10 * num)
        c = cv2.circle(color_transf, (x1+x2,y1+y2), 5, (255,0,0), -1)
        plt.imshow(c)

# plot the ma de
mx1 = int(4.25*swid/5) + 5
my1 = int(shei/27.5) + 4
for i, num in enumerate(cmd):
    if num != None:
        x2 = int(swid2/12 + swid2/6 * i)
        y2 = int(shei2/20 + shei2/10 * num)
        c = cv2.circle(color_transf, (mx1+x2,my1+y2), 5, (255,0,0), -1)
        plt.imshow(c)

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

def plot_ans(ans, x1, y1):
    for i, an in enumerate(ans):
        if an != None:
            y2 = int(ahei1/34 + ahei1/17 * i)
            x2 = int(awid1/8 + awid1/4 * ANS.index(an))
            c = cv2.circle(color_transf, (x1+x2,y1+y2), 5, (255,0,0), -1)
            plt.imshow(c)
answers = ['A', 'B', 'C','A', 'B', 
            'C','A', 'B', 'C','A', 
            'B', 'C', 'D', 'D', 'D',
            'A', 'B', 'C','A', 'B', 
            'C','A', 'B', 'C','A', 
            'B', 'C', 'D', 'D', 'D',
            None, None, None, None]
plot_ans(ans1, ax1, ay1)
plot_ans(ans2, a2x1, ay1)



students = ans1 + ans2
total = 0
points = 0
print('Grading\n')

for i in range(34):
    if answers[i] != None:
        print(f'Cau {i+1}: {students[i]}, {answers[i]}')
        if students[i] == answers[i]:
            points += 1
        total += 1
grade = round(points / total * 10,2)
xg = int(swid/10)
yg = int(shei/2.7)
g = cv2.putText(color_transf, str(grade), (xg,yg), cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 0 ,0), 5, cv2.LINE_AA)
plt.imshow(g)
plt.axis('off')
plt.savefig('graded.png', bbox_inches='tight', dpi=1200, transparent=True, pad_inches=0)
plt.show() 
