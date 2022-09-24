import sys
import numpy as np
import soundfile

def get_noise_info(noise_scp, noise_db_range):
    noises = []
    with open(noise_scp, "r", encoding="utf-8") as f:
        for line in f:
            sps = line.strip().split(None, 1)
            if len(sps) == 1:
                noises.append(sps[0])
            else:
                noises.append(sps[1])
    sps = noise_db_range.split("_")
    if len(sps) == 1:
        noise_db_low, noise_db_high = float(sps[0])
    elif len(sps) == 2:
        noise_db_low, noise_db_high = float(sps[0]), float(sps[1])
    else:
        raise ValueError("Format error: '{noise_db_range}' e.g. -3_4 -> [-3db,4db]")

    return noises, noise_db_low, noise_db_high

def write_to_scp(speech_scp, noises, noise_db_low, noise_db_high):
    with open(speech_scp, "r") as speech_scp_f:
        for line in speech_scp_f.readlines():
            if 0.875 < np.random.random():
                continue
            noise_path = np.random.choice(noises)
            noise_db = np.random.uniform(noise_db_low, noise_db_high)
            speech, _ = soundfile.read(line.split()[-1].strip(), always_2d=False)
            nsamples = len(speech)
            with soundfile.SoundFile(noise_path) as f:
                if f.frames == nsamples:
                    offset = 0
                elif f.frames < nsamples:
                    offset = np.random.randint(0, nsamples - f.frames)
                else:
                    offset = np.random.randint(0, f.frames - nsamples)
                    assert (f.frames - offset) >= nsamples

            print(line.split()[0].strip(), noise_path, noise_db, offset)

if __name__ == "__main__":
    noise_scp = sys.argv[1].strip()
    speech_scp = sys.argv[2].strip()
    noise_db_range = "0_25"
    noises, noise_db_low, noise_db_high = get_noise_info(noise_scp, noise_db_range)
    write_to_scp(speech_scp, noises, noise_db_low, noise_db_high)

