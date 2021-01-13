import napari
from pathlib import Path
from aicsimageio import AICSImage

#%%
data_root = Path("E:/DA/Data/scoss/Data/Jan2021/celltiffs")
file = Path(data_root / 'TUBA1B 243734 seg.tif')
#%%
seg = AICSImage(file).data.squeeze()
print(seg.shape)

#%%
with napari.gui_qt():
    viewer = napari.view_image(
        seg,
        channel_axis = 0
    )
