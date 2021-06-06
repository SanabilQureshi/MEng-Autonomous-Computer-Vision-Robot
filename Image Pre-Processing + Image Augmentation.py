import image_edit
import shutil
import glob
import os


Defects = ['Blobs','Burns','Crack']
# Defects = ['BMedm','BMega']
States = ['Defect','Non-Defect']

for Defect in Defects:
    for state in States:
        fdirs = [f'F:/### Sanabil Dissertation/Image Dataset/{Defect}/Pre-processing/Original',                             ## 0
                 f'F:/### Sanabil Dissertation/Image Dataset/{Defect}/Pre-processing/Converted',                            ## 1
                 f'F:/### Sanabil Dissertation/Image Dataset/{Defect}/Pre-processing/Cropped/{state}',                      ## 2
                 f'F:/### Sanabil Dissertation/Image Dataset/{Defect}/Pre-processing/Selected/{state}',                     ## 3
                 f'F:/### Sanabil Dissertation/Image Dataset/{Defect}/Post-Processing/Circled/Threshold',                   ## 4
                 f'F:/### Sanabil Dissertation/Image Dataset/{Defect}/Post-Processing/Circled/Grabcut',                     ## 5
                 f'F:/### Sanabil Dissertation/Image Dataset/{Defect}/Post-Processing/Augmentation/Flipped',                ## 6
                 f'F:/### Sanabil Dissertation/Image Dataset/{Defect}/Post-Processing/Augmentation/Rotated',                ## 7
                 f'F:/### Sanabil Dissertation/Image Dataset/{Defect}/Post-Processing/Augmentation/Noised',                 ## 8
                 f'F:/### Sanabil Dissertation/Image Dataset/{Defect}/Post-Processing/Result/Threshold',                    ## 9
                 f'F:/### Sanabil Dissertation/Image Dataset/{Defect}/Post-Processing/Result/Grabcut',                      ## 10
                 f'F:/### Sanabil Dissertation/Image Dataset/{Defect}/Learning/Test/{state}',                               ## 11
                 f'F:/### Sanabil Dissertation/Image Dataset/{Defect}/Learning/Train/{state}']                              ## 12

        Remove_list = [fdirs[3], fdirs[9], fdirs[10], fdirs[11], fdirs[12]]

        dataset_split = 0.8

        for item in Remove_list:
            image_edit.Remover(item)

        image_edit.Renamer(fdirs[2])
        image_edit.Splitter(fdirs[2], fdirs[3], dataset_split)
        image_edit.Splitter(fdirs[2], fdirs[11], 1 - dataset_split)

        image_edit.Any2PNG(fdirs[0], fdirs[1])                                                  ## Original > Converted
        image_edit.Cropper(fdirs[1], fdirs[2])                                                  ## Converted > Cropped

        # Morph-cut

        image_edit.Circler(fdirs[3], fdirs[4], transparent=False)                                ## Cropped > Circled
        image_edit.Flipper(fdirs[4], fdirs[6], direction = 'both')                               ## Circled > Flipped
        image_edit.Rotator(fdirs[4], fdirs[7], step_size=36)                                     ## Circled > Rotated
        # image_edit.Noisy(fdirs[5], fdirs[9])                                                   ## Circled > Noised

        Threshlist = [fdirs[4],fdirs[6],fdirs[7],fdirs[8]]

        for item in Threshlist:
            Results = glob.glob(str(f'{item}' + "\*.png"))
            for file in Results:
                shutil.move(file, fdirs[12])

        # Grab-cut

        image_edit.Grabcutter(fdirs[3], fdirs[5], iterations=1, transparent=False)               ## Cropped > Circled
        image_edit.Flipper(fdirs[5], fdirs[6], direction='both')                                 ## Circled > Flipped
        image_edit.Rotator(fdirs[5], fdirs[7], step_size=36)                                     ## Circled > Rotated
        # image_edit.Noisy(fdirs[4], fdirs[7])                                                   ## Circled > Noised

        Grablist = [fdirs[5],fdirs[6],fdirs[7],fdirs[8]]

        for item in Grablist:
            Results = glob.glob(str(f'{item}' + "\*.png"))
            for file in Results:
                shutil.move(file, fdirs[10])

    image_edit.Remover(fdirs[12])

# image_edit.Padder(fdirs[2], fdirs[5], padding = 5000)                                   ## Padded In  > Padded Out
# image_edit.Transparenter([unit_test[0]], unit_test[1])                                  ## Transparent In > Transparent Out
# image_edit.Rescaler([unit_test[2]], unit_test[3], scale_factor=0.5)                     ## Rescaled In > Rescaled Out