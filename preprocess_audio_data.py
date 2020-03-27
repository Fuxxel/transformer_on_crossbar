import librosa
import numpy as np
import h5py

def filter_db(sample, top_db=60):
    # Based on librosa reference

    mse = librosa.feature.rms(y=sample) ** 2
    mse_db = librosa.core.power_to_db(mse.squeeze()) > top_db

    indices = np.nonzero(mse_db)[0]

    if len(indices) > 0:
        start = librosa.core.frames_to_samples(indices[0])
        end = min(sample.shape[-1], librosa.core.frames_to_samples(indices[-1]))
    else:
        start, end = 0, 0

    return sample[start:end], (start, end)

def main():
    source_path = "data/all_coins.hdf5"
    destination_path = "data/all_coins_preprocessed.hdf5"

    source = h5py.File(source_path,"r")
    destination = h5py.File(destination_path,"w")

    for coin in source.keys():
        sample_number = 0
        destination.create_group(coin)
        for g in source[coin].keys():
            for i in source[coin][g].keys():
                print(f"\r{sample_number + 1}", end="")
                timeseries = source[coin][g][i]["values"][()]
                trimmed_timeseries, _ = filter_db(timeseries)

                destination[coin].create_dataset(str(sample_number), data=trimmed_timeseries, compression="gzip")                
                sample_number += 1
        print()
                
    destination.close()
    source.close()

if __name__ == "__main__":
    main()