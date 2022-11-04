import pynput

def lock(aims, mouse, x, y):
    mouse_pos_x, mouse_pos_y = mouse.position
    dist_list = []
    for det in aims:
        _, x_c, y_c, _, _ = det
        dist = (x * float(x_c) - mouse_pos_x) ** 2 + (y * float(y_c) - mouse_pos_y) ** 2
        dist_list.append(dist)

    index = dist_list.index(min(dist_list))
    det = aims[index]

    tag, x_center, y_center, width, height = det
    tag = int(tag)
    x_center, width = x * float(x_center) // 1.25, x * float(width) // 1.25
    y_center, height = y * float(y_center) // 1.25, y * float(height) // 1.25
    # x_center, width = x * float(x_center), x * float(width)
    # y_center, height = y * float(y_center), y * float(height)

    if tag == 0 or tag == 2:
        mouse.position = (x_center , y_center)
    elif tag == 1 or tag == 3:
        mouse.position = (x_center, y_center - 1 / 6 * height)