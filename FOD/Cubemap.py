import pdb
from icecream import ic
from collections import defaultdict
import numpy as np
class CubemapHandler():
    def __init__(self, keyword):
        self.keyword = keyword
        
    def findCubemapFiles(self, filenames):
        self.cubemap_files = []
        self.idxs = []
        ic(filenames)
        self.dict = defaultdict(list)
        for idx, filename in enumerate(filenames):
            # print(filename[:7], self.keyword)
            if filename[:7] == self.keyword:
                self.cubemap_files.append(filename)
                self.idxs.append(idx)
                filename_id = self.getFilenameID(filename)
                self.dict[filename_id].append(idx)


        ic(self.cubemap_files)
        ic(self.dict)
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
        query_image_names_not_reduced = []
        for file in query_image_names:
            # print(file)
            filename_ID = self.getFilenameID(file)
            # print(filename_ID)
            cubemap_filenames = [i for i in paths_images_not_reduced if filename_ID in i]
            # print(cubemap_filenames)
            query_image_names_not_reduced.extend(cubemap_filenames)
        print(query_image_names_not_reduced)
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

