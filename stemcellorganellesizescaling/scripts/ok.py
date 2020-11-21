# %%
data_root = Path(f"E:/DA/Data/scoss/Data/Subsample_Nov2020/{datastr}")
pic_root =  Path(f"E:/DA/Data/scoss/Pics/Subsample_Nov2020/{datastr}")
org_root =  Path('E:/DA/Data/scoss/Data/Nov2020')
data_root.mkdir(exist_ok=True)
pic_root.mkdir(exist_ok=True)
copyfile((org_root / 'SizeScaling_20201102.csv'), (data_root / 'SizeScaling_20201102.csv'))


