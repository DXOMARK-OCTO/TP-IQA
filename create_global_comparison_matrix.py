# pylint: disable=wrong-import-position

import os
import numpy as np
from lockfile import FileLocker

def computeCommonCompDict(resultsPath, dataPath, experiment):
    '''
    Sum all individuals results to compute the common matrix
    '''
    compDict = {}
    commonCompMat = None
    listFiles = None
    for file in os.listdir(resultsPath):
        if "compMat" in file and os.path.splitext(file)[-1] == ".npz": # TODO: implement a better test (robust regex)

            userMatrix = os.path.join(resultsPath, file)
            userMatrixLocker = FileLocker(userMatrix)
            userMatrixLocker.lock()
            try:
                with np.load(userMatrix, mmap_mode="r") as compMatData:

                    if not ("COMP_MAT" in compMatData and "LIST_FILES" in compMatData):
                        print(f"User matrix looks incomplete: {userMatrix}")
                        continue

                    compMatUser = compMatData["COMP_MAT"]
                    # Check if are the same and well ordered.
                    if "LIST_FILES" in compMatData:
                        listFilesForUser = compMatData["LIST_FILES"]
                        listFiles = listFiles if listFiles is not None else listFilesForUser
                        if not (listFilesForUser == listFiles).all():
                            if set(listFilesForUser) != set(listFiles):
                                raise ValueError(f'The list of images for {experiment} is different from the one used in previous iterations')
                            # in this case, we have the same images, but the order is different. We need to adapt the compMatData to follow the same order.
                            newOrder = [np.where(listFilesForUser==value)[0][0] for value in listFiles]
                            compMatUser = compMatUser[:, newOrder][newOrder]
                    if commonCompMat is not None:
                        commonCompMat = np.add(commonCompMat, compMatUser)
                    else:
                        commonCompMat = compMatUser
            except Exception as ex:
                print(f"Error reading user matrix {userMatrix}. {ex}")
            userMatrixLocker.unlock()

    if commonCompMat is not None:
        compDict['COMP_MAT'] = commonCompMat
        compDict['PROCESSED_COMPARISONS'] = np.sum(commonCompMat)
        compDict['LIST_FILES'] = listFiles if listFiles is not None else sorted([img for img in os.listdir(dataPath) if img.lower().endswith(EXTENSIONS)])

    return compDict
