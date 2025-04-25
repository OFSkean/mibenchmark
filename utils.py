import numpy as np
import torchvision

def nat2bit(nat):
    return nat * np.log2(np.exp(1))

def bit2nat(bit, steps=0):
    def b2n(v):
        return v / np.log2(np.exp(1))
    
    if steps == 0:
        return b2n(bit)
    elif isinstance(bit, list):
        _ = steps // len(bit)
        t = []
        for tmi in bit:
            t += [b2n(tmi)] * _
        return t
    else:
        return [b2n(bit)] * steps
        
def cal_bsc(patches, target_mi): #target_mi in bit scale
    if np.prod(patches) == target_mi:
        print("NO BSC")
        return 0
    else:
        x = np.arange(0, 0.501, 0.0001)
        mi = patches * (1 + (x * np.log2(x) + (1-x) * np.log2(1-x)))
        mi[0] = patches # fix nan error
        assert np.all(np.isfinite(mi))

        truth = np.full(len(x), target_mi)

        result = x[np.argmin(np.abs(truth - mi))]
        return result