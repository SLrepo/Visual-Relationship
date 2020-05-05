import cv2
WIN_MAX_SIZE = 1000


def print_xy(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y, " : ", param[y, x])


def display_image(win_name, image):
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)  # Create window; allow # resizing
    h = image.shape[0]  # image height
    w = image.shape[1]  # image width
    # Shrink the window if it is too big(exceeds some maximum size).
    if max(w, h) > WIN_MAX_SIZE:
        scale = WIN_MAX_SIZE / max(w, h)
    else:
        scale = 1
    cv2.resizeWindow(winname=win_name, width=int(w * scale), height=int(h * scale))
    # Assign callback function, and show
    cv2.setMouseCallback(window_name=win_name, on_mouse=print_xy, param=image)
    cv2.imshow(win_name, image)


def main():
    bgr_image = cv2.imread("new_data/jaco.png")
    height = bgr_image.shape[0]
    width = bgr_image.shape[1]
    print("Size of this image: (width,height): (%d,%d)" % (width, height))
    cv2.namedWindow("my image", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name="my image", on_mouse=print_xy, param=bgr_image)
    cv2.imshow("my image", bgr_image)
    cv2.waitKey(0)
    bgr_image2 = cv2.cvtColor(src=bgr_image,  code=cv2.COLOR_BGR2GRAY)
    cv2.imshow("Grey image", bgr_image2)
    cv2.waitKey(0)
    display_image("new image", bgr_image)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()