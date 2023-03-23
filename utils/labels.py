VINDR_CXR_LABELS = {
    'No finding': 'no finding',
    'Other diseases': 'other diseases',
    'Aortic enlargement': 'aortic enlargement',
    'Cardiomegaly': 'cardiomegaly',
    'Pulmonary fibrosis': 'pulmonary fibrosis',
    'Pleural thickening': 'pleural thickening',
    'Pleural effusion': 'pleural effusion',
    'Lung Opacity': 'lung opacity',
    'Other lesion': 'other lesion',
    'Pneumonia': 'pneumonia',
    'Tuberculosis': 'tuberculosis',
    'Nodule/Mass': 'nodule or mass',
    'Infiltration': 'infiltration',
    'Calcification': 'calcification',
    'ILD': 'interstitial lung disease',
    'Consolidation': 'consolidation',
    'Lung tumor': 'lung tumor',
    'Mediastinal shift': 'mediastinal shift',
    'Atelectasis': 'atelectasis',
    'Pneumothorax': 'pneumothorax',
    'Enlarged PA': 'enlarged pulmonary artery',
    'Rib fracture': 'rib fracture',
    'Emphysema': 'emphysema',
    'Lung cavity': 'lung cavity',
    'COPD': 'chronic obstructive pulmonary disease',
    'Lung cyst': 'lung cyst',
    'Clavicle fracture': 'clavicle fracture',
    'Edema': 'edema'
}

MIMIC_CXR_LABELS = {
    'No Finding': 'no Finding',
    'Support Devices': 'support devices',
    'Pleural Effusion': 'pleural effusion',
    'Lung Opacity': 'lung opacity',
    'Atelectasis': 'atelectasis',
    'Cardiomegaly': 'cardiomegaly',
    'Edema': 'edema',
    'Pneumonia': 'pneumonia',
    'Consolidation': 'consolidation',
    'Pneumothorax': 'pneumothorax',
    'Enlarged Cardiomediastinum': 'enlarged cardiomediastinum',
    'Lung Lesion': 'lung lesion',
    'Fracture': 'fracture',
    'Pleural Other': 'pleural other'
}

MURED_LABELS = {
    'DR': 'diabetic retinopathy',
    'NORMAL': 'normal retina',
    'MH': 'media haze',
    'ODC': 'optic disc cupping',
    'TSLN': 'tessellation',
    'ARMD': 'age-related macular degeneration',
    'DN': 'drusen',
    'MYA': 'myopia',
    'BRVO': 'branch retinal vein occlusion',
    'ODP': 'optic disc pallor',
    'CRVO': 'central retinal vein oclussion',
    'CNV': 'choroidal neovascularization',
    'RS': 'retinitis',
    'ODE': 'optic disc edema',
    'LS': 'laser scars',
    'CSR': 'central serous retinopathy',
    'HTR': 'hypertensive retinopathy',
    'ASR': 'arteriosclerotic retinopathy',
    'CRS': 'chorioretinitis',
    'OTHER': 'other diseases'
}

# {'No finding': 12630, 'Other diseases': 3184, 'Aortic enlargement': 2249, 'Cardiomegaly': 1913, 'Pleural thickening': 1271, 'Pulmonary fibrosis': 991, 'Lung Opacity': 697, 'Other lesion': 572, 'Pneumonia': 604, 'Pleural effusion': 607, 'Tuberculosis': 429, 'Infiltration': 395, 'ILD': 349, 'Consolidation': 200}
VINDR_SPLIT = {
    'train': [
        'No finding', 'Other diseases', 'Aortic enlargement', 'Cardiomegaly', 
        'Pleural thickening','Pulmonary fibrosis', 'Lung Opacity', 'Other lesion',
        'Pneumonia','Pleural effusion','Tuberculosis',
        'Infiltration','ILD','Consolidation'
    ],
    'train-subset': [
        # 'No finding', 'Other diseases', 
        'Aortic enlargement', 'Cardiomegaly', 
        'Pleural thickening','Pulmonary fibrosis', 'Lung Opacity',
        'Other lesion',
        'Pneumonia',
        # 'Pleural effusion','Tuberculosis',
        # 'Infiltration',
        # 'ILD','Consolidation'
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