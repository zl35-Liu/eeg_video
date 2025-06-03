import numpy as np
from DE_PSD import DE_PSD  #差分熵和功率谱密度
from tqdm import tqdm

# Extract DE or PSD features with a 2-second window, that is, for each 2-second EEG segment, we extract a DE or PSD feature.
# Input the shape of (7 * 40 * 5 * 62 * 2s*fre), meaning 7 blocks, 40 concepts, 5 video clips, 62 channels, and 2s*fre time-points.
# Output the DE or PSD feature with (7 * 40 * 5 * 62 * 5), the last 5 indicates the frequency bands' number.

fre = 200

for subname in range(1,21):

    loaded_data = np.load('data/EEG2Video/Segmented_Rawf_200Hz_2s/sub'+ str(subname) + '.npy')
    # (7 * 40 * 5 * 62 * 2*fre)

    print("Successfully loaded .npy file.")
    print("Loaded data:")

    DE_data = np.empty((0, 40, 5, 62, 5))
    PSD_data = np.empty((0, 40, 5, 62, 5))

    for block_id in range(7):
        print("block: ", block_id)
        now_data = loaded_data[block_id]
        de_block_data = np.empty((0, 5, 62, 5))
        psd_block_data = np.empty((0, 5, 62, 5))
        for class_id in tqdm(range(40)):
            de_class_data = np.empty((0, 62, 5))
            psd_class_data = np.empty((0, 62, 5))
            for i in range(5):
                de, psd = DE_PSD(now_data[class_id, i, :, :].reshape(62, 2*fre), fre, 2)
                de_class_data = np.concatenate((de_class_data, de.reshape(1, 62, 5)))
                psd_class_data = np.concatenate((psd_class_data, psd.reshape(1, 62, 5)))
            de_block_data = np.concatenate((de_block_data, de_class_data.reshape(1, 5, 62, 5)))
            psd_block_data = np.concatenate((psd_block_data, psd_class_data.reshape(1, 5, 62, 5)))
        DE_data = np.concatenate((DE_data, de_block_data.reshape(1, 40, 5, 62, 5)))
        PSD_data = np.concatenate((PSD_data, psd_block_data.reshape(1, 40, 5, 62, 5)))

    np.save("data/EEG2Video/DE_1per2s/" + subname +".npy", DE_data)
    np.save("data/EEG2Video/PSD_1per2s/" + subname + ".npy", PSD_data)