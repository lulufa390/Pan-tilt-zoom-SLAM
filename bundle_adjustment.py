

class BundleAdjust:
    def __init__(self, annotation_path, image_path):
        seq = sio.loadmat(annotation_path)
        self.annotation = seq["annotation"]
        self.meta = seq['meta']

        self.u, self.v = self.annotation[0][0]['camera'][0][0:2]
        self.base_rotation = np.zeros([3, 3])
        cv.Rodrigues(self.meta[0][0]["base_rotation"][0], self.base_rotation)
        self.c = self.meta[0][0]["cc"][0]

        self.image_path = image_path

    def get_image_gray(self, index):
        """
        :param index: image index for basketball sequence
        :return: gray image
        """
        # img = cv.imread(self.image_path + "00000" + str(index + 515) + ".jpg")
        img = cv.imread(self.image_path + self.annotation[0][index]['image_name'][0])

        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        return img_gray