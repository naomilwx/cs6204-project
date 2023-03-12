VINDR_CXR_LABELS = {
    'No finding': 'No finding',
    'Other diseases': 'Other diseases',
    'Aortic enlargement': 'Aortic enlargement',
    'Cardiomegaly': 'Cardiomegaly',
    'Pulmonary fibrosis': 'Pulmonary fibrosis',
    'Pleural thickening': 'Pleural thickening',
    'Pleural effusion': 'Pleural effusion',
    'Lung Opacity': 'Lung Opacity',
    'Other lesion': 'Other lesion',
    'Pneumonia': 'Pneumonia',
    'Tuberculosis': 'Tuberculosis',
    'Nodule/Mass': 'Nodule or Mass',
    'Infiltration': 'Infiltration',
    'Calcification': 'Calcification',
    'ILD': 'Interstitial lung disease',
    'Consolidation': 'Consolidation',
    'Lung tumor': 'Lung tumor',
    'Mediastinal shift': 'Mediastinal shift',
    'Atelectasis': 'Atelectasis',
    'Pneumothorax': 'Pneumothorax',
    'Enlarged PA': 'Enlarged PA',
    'Rib fracture': 'Rib fracture',
    'Emphysema': 'Emphysema',
    'Lung cavity': 'Lung cavity',
    'COPD': 'Chronic obstructive pulmonary disease',
    'Lung cyst': 'Lung cyst',
    'Clavicle fracture': 'Clavicle fracture',
    'Edema': 'Edema'
}

MIMIC_CXR_LABELS = {
    'No Finding': 'No Finding',
    'Support Devices': 'Support Devices',
    'Pleural Effusion': 'Pleural Effusion',
    'Lung Opacity': 'Lung Opacity',
    'Atelectasis': 'Atelectasis',
    'Cardiomegaly': 'Cardiomegaly',
    'Edema': 'Edema',
    'Pneumonia': 'Pneumonia',
    'Consolidation': 'Consolidation',
    'Pneumothorax': 'Pneumothorax',
    'Enlarged Cardiomediastinum': 'Enlarged Cardiomediastinum',
    'Lung Lesion': 'Lung Lesion',
    'Fracture': 'Fracture',
    'Pleural Other': 'Pleural Other'
}

MURED_LABELS = {
    'DR': 'Diabetic Retinopathy',
    'NORMAL': 'Normal Retina',
    'MH': 'Media Haze',
    'ODC': 'Optic Disc Cupping',
    'TSLN': 'Tessellation',
    'ARMD': 'Age-Related Macular Degeneration',
    'DN': 'Drusen',
    'MYA': 'Myopia',
    'BRVO': 'Branch Retinal Vein Occlusion',
    'ODP': 'Optic Disc Pallor',
    'CRVO': 'Central Retinal Vein Oclussion',
    'CNV': 'Choroidal Neovascularization',
    'RS': 'Retinitis',
    'ODE': 'Optic Disc Edema',
    'LS': 'Laser Scars',
    'CSR': 'Central Serous Retinopathy',
    'HTR': 'Hypertensive Retinopathy',
    'ASR': 'Arteriosclerotic Retinopathy',
    'CRS': 'Chorioretinitis',
    'OTHER': 'Other Diseases'
}

VINDR_SPLIT = {
    'train': [
        'No finding', 'Other diseases', 'Aortic enlargement', 'Cardiomegaly', 
        'Pleural thickening','Pulmonary fibrosis', 'Lung Opacity', 'Other lesion',
        'Pneumonia','Pleural effusion','Tuberculosis',
        'Infiltration','ILD','Consolidation'
    ],
    'test': [
        'Lung tumor','Nodule/Mass','Edema','Lung cyst',
        'Rib fracture','Clavicle fracture','Lung cavity'
    ],
    'val': [
        'COPD','Pneumothorax','Mediastinal shift', 'Emphysema',
        'Enlarged PA', 'Atelectasis', 'Calcification'
    ]
}

MURED_SPLIT = {
    'train': [
      'DR', 'NORMAL','ODC','OTHER','MH',
      'DN','ARMD','TSLN','MYA','CNV'
    ],
    'test': [
      'BRVO','CSR','HTR','ASR','ODE'
    ],
    'val': [
       'RS','CRS','LS','ODP','CRVO'
    ]
}