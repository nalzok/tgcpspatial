from config import datadir
from lgcp.data import Dataset
from lgcp.infer import lgcp2d
from lgcp.kern import kernelft

fn = "r2405_051216b_cell1816.mat"
data = Dataset.from_file(datadir + fn).prepare()
arena = data.arena

# Used to locate peaks in the autocorrelogram
kf = kernelft(data.shape, data.P, data.V, angle=data.angle, style="grid")
result, model = lgcp2d(
    kf, data.N, data.K, data.prior_mean, (data.kdelograte, None), eps=1e-5, verbose=True
)

arena.imshow(result.info.r, lw=8)
print(data.arena.meters_per_bin * 100)
