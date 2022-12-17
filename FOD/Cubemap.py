import pdb
from icecream import ic
from collections import defaultdict
import numpy as np
class CubemapHandler():
    def __init__(self, keyword):
        self.keyword = keyword
    def checkCubemapFace(self, filename):
        if filename[:7] == self.keyword:
            return True
        else:
            return False
    def findCubemapFiles(self, filenames):
        self.cubemap_files = []
        self.idxs = []
        # ic(filenames)
        self.dict = defaultdict(list)
        for idx, filename in enumerate(filenames):
            # print(filename[:7], self.keyword)
            if self.checkCubemapFace(filename):
                self.cubemap_files.append(filename)
                self.idxs.append(idx)
                filename_id = self.getFilenameID(filename)
                self.dict[filename_id].append(idx)


        # ic(self.cubemap_files)
        # ic(self.dict)
    def getFilenameID(self, filename):
        filename_id = filename.split('.')[0].split('_')[1]
        return filename_id

    def reduceArray(self, array):
        mask = np.ones((array.shape[0]))
        for key, value in self.dict.items():
            # pdb.set_trace()
            array[value[0]] = np.mean(array[value], axis=0)
            for value_element in value[1:]:
                mask[value_element] = 0
        
        return array[mask == 1], mask

    def reduceArray(self, array):
        mask = np.ones((array.shape[0]))
        for key, value in self.dict.items():
            # pdb.set_trace()
            array[value[0]] = np.mean(array[value], axis=0)
            for value_element in value[1:]:
                mask[value_element] = 0
        
        return array[mask == 1], mask

    def reduceFilenames(self, filenames):
        mask = np.ones((filenames.shape[0]))
        for key, value in self.dict.items():
            # pdb.set_trace()
            for value_element in value[1:]:
                mask[value_element] = 0
        
        return filenames[mask == 1]

        # pdb.set_trace()

    def getCubemapFilenamesFromSingleFaceNames(self,
        paths_images_not_reduced,
        query_image_names):

        paths_images_not_reduced = [x.split('\\')[-1] for x in paths_images_not_reduced]

        query_image_names_not_reduced = []
        for filename in query_image_names:
            print(filename)
            if self.checkCubemapFace(filename):
                filename_ID = self.getFilenameID(filename)
                # print(filename_ID)
                cubemap_filenames = [i for i in paths_images_not_reduced if filename_ID in i]
                # print(cubemap_filenames)
                query_image_names_not_reduced.extend(cubemap_filenames)
            # else:
            #     query_image_names_not_reduced.extend(filename)
        print(len(query_image_names_not_reduced))

        return query_image_names_not_reduced

    # def findUniqueIdxsInCubemapFiles(self):
    #     for file in self.cubemap_files:




    # def getCubemapDict(self):
    #      = get

    '''
    self.dict = {'0asef0saef': [91, 49, 100, 212],
    '0asef0saef': [91, 49, 100, 212],}
    '''
    # def reduceCubemapValues(self, array):
        # array is a NPY array    
    #     for cubemap_file in self.cubemap_files:
    #         array[idx]

