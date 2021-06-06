import cv2
import glob
import image_edit
from silence_tensorflow import silence_tensorflow
import numpy as np
import random
from pathlib import Path
import pathlib
import pandas as pd

silence_tensorflow()

import GPU_RESET
import tensorflow as tf

GPU_RESET.reset_keras()

index = []
columns = []

# df = pd.DataFrame(index=index, columns=columns)
# df = df.fillna(0) # with 0s rather than NaNs

# Defects = ['BMedm','BMega']
Defects = ['Blobs', 'Burns', 'Crack']

# Defects = ['Blobs']

CATEGORIES = ["Defect", "Non-Defect"]

for Defect in Defects:
    TestDir_Defect = f'F:/### Sanabil Dissertation/Image Dataset/{Defect}/Learning/Test/Defect'
    TestDir_Non_Defect = f'F:/### Sanabil Dissertation/Image Dataset/{Defect}/Learning/Test/Non-Defect'

    Models = glob.glob(f'F:/### Sanabil Dissertation/Disso_Scripts/Main_Code/models/{Defect}/*')
    # print(Models)
    TestImages = glob.glob(str(f'{TestDir_Defect}' + "\*.png")) + glob.glob(str(f'{TestDir_Non_Defect}' + "\*.png"))
    # random.shuffle(TestImages)

    Defect_files = glob.glob(str(f'{TestDir_Defect}' + "/*.png"))
    Non_Defect_files = glob.glob(str(f'{TestDir_Non_Defect}' + "/*.png"))

    img_list = []
    test_status = []

    for img in Defect_files:
        img = img[69:]
        img_list.append(img)
        test_status.append('Defect')

    for img in Non_Defect_files:
        img = img[73:]
        img_list.append(img)
        test_status.append('Non-Defect')

    current_col = 1

    df = pd.DataFrame({"Image list": img_list,
                       "Test Status": test_status})
    final_list = ['Overall Accuracy', '1']

    start_array = []

    for trained_model in Models:
        current_col += 1
        model = tf.keras.models.load_model(trained_model)
        defect_list = []
        non_defect_list = []
        result_list = []

        modelname = str(trained_model[65:]).replace(" ", "+")
        modelsavename = str(trained_model[65:])

        log_location = f'{trained_model}/{trained_model[65:-6]}.log'
        column_header = trained_model[65:-31]
        print(log_location)
        print(column_header)

        for img in TestImages:
            prediction = model.predict([image_edit.image_input(img)])
            # print(f'{Defect} - ' + f'{trained_model[65:-31]} - ' + CATEGORIES[int(prediction[0][0])])
            result_list.append(CATEGORIES[int(prediction[0][0])])

        for img in Defect_files:
            prediction = model.predict([image_edit.image_input(img)])
            # print(f'{Defect} - ' + f'{trained_model[65:-31]} - ' + CATEGORIES[int(prediction[0][0])])
            if CATEGORIES[int(prediction[0][0])] == 'Defect':
                defect_list.append(1)
            else:
                defect_list.append(0)

            total_correct_defect = np.sum(defect_list)

        for img in Non_Defect_files:
            prediction = model.predict([image_edit.image_input(img)])
            # print(f'{Defect} - ' + f'{trained_model[65:-31]} - ' + CATEGORIES[int(prediction[0][0])])
            if CATEGORIES[int(prediction[0][0])] == 'Non-Defect':
                non_defect_list.append(1)
            else:
                non_defect_list.append(0)

            total_correct_non_defect = np.sum(non_defect_list)

        # print(len(status))
        final_result = (total_correct_defect + total_correct_non_defect) / len(TestImages)
        final_list.append(final_result)
        df.insert(loc=current_col, column=f'{column_header}', value=result_list)

        df2 = pd.read_csv(f"F:/### Sanabil Dissertation/Disso_Scripts/Main_Code/model2csv/Tensorboard/{Defect}/{modelsavename}.csv")

        df2.index.names = ['Date']
        del df2['Step']
        df2.columns.values[0] = "Total Time"
        df2.columns.values[1] = "Training Accuracy"
        df2['Epoch Duration'] = df2.xs('Total Time', axis=1).diff()
        df2['Epoch Duration'] = df2['Epoch Duration'].replace(np.nan, 0)
        df2['Total Time'] = df2['Epoch Duration'].cumsum()
        df2 = df2[['Total Time', 'Epoch Duration', 'Training Accuracy']]

        df2.insert(loc=0, column='Name', value=modelsavename)
        df2.insert(loc=1, column='Conv_layers', value=modelsavename[6:7])

        conv_string = '128'
        optimiser1 = 'adam'
        optimiser2 = 'RMSprop'
        optimiser3 = 'SGD'

        if conv_string in modelsavename:
            nodes = modelsavename[13:16]
            dense_layers = modelsavename[23:24]

            if optimiser1 in modelsavename:
                optimiser = modelsavename[31:35]
            elif optimiser2 in modelsavename:
                optimiser = modelsavename[31:38]
            else:
                optimiser = modelsavename[31:34]

        else:
            nodes = modelsavename[13:15]
            dense_layers = modelsavename[22:23]

            if optimiser1 in modelsavename:
                optimiser = modelsavename[31:35]
            elif optimiser2 in modelsavename:
                optimiser = modelsavename[30:34]
            else:
                optimiser = modelsavename[30:33]

        df2.insert(loc=2, column='Conv_nodes', value=nodes)
        df2.insert(loc=3, column='Dense_layers', value=dense_layers)
        df2.insert(loc=4, column='Dense_nodes', value=nodes)
        df2.insert(loc=5, column='Optimiser', value=optimiser)

        epochnumber = np.arange(len(df2))
        epochnumber += 1

        df2.insert(loc=6, column='Epoch Number', value=epochnumber)
        df2.insert(loc=10, column='Test Accuracy', value=final_result)

        df3 = pd.read_csv(log_location)
        del df3['epoch']
        del df3['accuracy']
        del df3['mean_absolute_percentage_error']
        del df3['val_mean_absolute_percentage_error']
        df3 = df3.rename(columns={'loss': 'Training Loss',
                                  'val_loss': 'Validation Loss',
                                  'mean_absolute_error': 'Mean Training Error',
                                  'val_mean_absolute_error': 'Mean Validation Error',
                                  'val_accuracy': 'Validation Accuracy'})

        df_final = pd.concat([df2, df3], axis=1)

        df_final.append(pd.Series(), ignore_index=True)
        df_final = df_final.replace(np.nan, ' ', regex=True)
        df_final = df_final[[col for col in df_final.columns if col != 'Test Accuracy'] + ['Test Accuracy']]
        start_array.append(df_final)
        # print(df_final.to_string())

        # print(df3.to_string())
    df5 = pd.concat(start_array)
    new_row = pd.Series(final_list, index=df.columns)
    df = df.append(new_row, ignore_index=True)

    df.to_csv(f'F:/### Sanabil Dissertation/Disso_Scripts/Main_Code/model2csv/{Defect}/Test_Data_results_{Defect}.csv')
    df5.to_csv(f'F:/### Sanabil Dissertation/Disso_Scripts/Main_Code/model2csv/{Defect}/Performance_Data_results_{Defect}.csv')

    # print(df)
