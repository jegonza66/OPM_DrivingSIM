import os
import shutil
import osl_dynamics
import paths
import nibabel as nib
import numpy as np
import json

 
################ DOWNLOAD PARCELLATION FILE ################
print(f'\n \033[96mCopiando parcellation file desde osl_dynamics al directorio atlas ... ')
src_parc = os.path.join(
    os.path.dirname(osl_dynamics.__file__),
    'files', 'parcellation',
    'fmri_d100_parcellation_with_PCC_reduced_2mm_ss5mm_ds8mm.nii.gz'
)

dest_parc = os.path.join(paths.atlas_path, 'fmri_d100_parcellation_with_PCC_reduced_2mm_ss5mm_ds8mm.nii.gz')
os.makedirs(os.path.dirname(dest_parc), exist_ok=True)

shutil.copy2(src_parc, dest_parc)
print(f'\n \033[96mCopiado a: {dest_parc} ')


################ CREATING PARCEL-REGION MAPPING ################
print(f'\n \033[96mCreando mapa de parcelas a regiones ... ')
atlas_img = nib.load(dest_parc)
atlas_data = atlas_img.get_fdata()

n_parcels = atlas_data.shape[3]
parcel_ids = list(range(1, n_parcels + 1))
occipital_parcels = list(range(int(n_parcels * 0.75), n_parcels))

atlas_mapping = {
    "parcel_ids": parcel_ids,
    "occipital": occipital_parcels
}

mapping_path = os.path.join(paths.atlas_path, 'fmri_d100_parcellation_mapping.json')

with open(mapping_path, 'w') as f:
    json.dump(atlas_mapping, f, indent=2)

print(f'Mapping guardado en {mapping_path}')


################ DOWNLOAD MASK FILE ################
# Copy mask file
print(f'\n \033[96mCopiando mask file desde osl_dynamics al directorio atlas ... ')
src_mask = os.path.join(
    os.path.dirname(osl_dynamics.__file__),
    'files', 'mask',
    'MNI152_T1_8mm_brain.nii.gz'
)
dest_mask = os.path.join(paths.atlas_path, 'MNI152_T1_8mm_brain.nii.gz')
shutil.copy2(src_mask, dest_mask)
print(f'\n \033[96mCopiado a: {dest_mask} ')