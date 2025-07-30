import nibabel as nib
import os

input_folder = "D:/New Streamlit"
output_folder = "D:/New Streamlit/cut_30slice"
os.makedirs(output_folder, exist_ok=True)

for i in range(1, 21):  # File 01.nii.gz sampai 20.nii.gz
    fname = f"{i:02d}.nii.gz"
    input_path = os.path.join(input_folder, fname)
    output_path = os.path.join(output_folder, f"{i:02d}_mid30.nii.gz")

    if not os.path.exists(input_path):
        print(f"File tidak ditemukan: {input_path}")
        continue

    print(f"Memotong: {fname}")
    img = nib.load(input_path)
    data = img.get_fdata()
    affine = img.affine
    header = img.header

    z = data.shape[2]
    if z < 30:
        print(f"Slice tidak cukup ({z}) di file {fname}, dilewati.")
        continue

    # Ambil 30 slice dari tengah
    start = z // 2 - 15
    end = z // 2 + 15
    data_cut = data[:, :, start:end]
    header.set_data_shape(data_cut.shape)

    # Simpan
    img_cut = nib.Nifti1Image(data_cut, affine, header)
    nib.save(img_cut, output_path)
    print(f"Disimpan: {output_path} | Shape: {data_cut.shape}")
