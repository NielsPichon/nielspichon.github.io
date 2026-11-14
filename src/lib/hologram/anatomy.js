// Structure catalogue for the radiotherapy hologram.
//
// Meshes come from the Human Reference Atlas 3D reference organs (HuBMAP / NIH, CC BY 4.0,
// https://humanatlas.io/3d-reference-library), merged into radiotherapy structures, remeshed and
// decimated in Blender, then Draco-compressed. The thyroid, oesophagus, stomach, right common carotid and lymph node
// stations are not in the atlas and were modelled around its landmarks.
// Each GLB node is named after a structure `id` below; `skin` and `skeleton` form the body shell.

export const BODY_COLOR = '#4cc9ff';
export const IDLE_COLOR = '#2f86b0';

export const MODEL = '/models/anatomy-male.glb';

export const ATTRIBUTION = {
  label: 'Anatomy: Human Reference Atlas (CC BY 4.0), modified',
  href: 'https://humanatlas.io/3d-reference-library'
};

/** Radiotherapy regions of interest; the camera frames each from its structures' bounds. */
export const GROUPS = [
  { id: 'hn', name: 'Head & Neck', color: '#ffae42' },
  { id: 'chest', name: 'Chest', color: '#4dffc3' },
  { id: 'heart', name: 'Heart', color: '#ff4f7b' },
  { id: 'abdomen', name: 'Abdomen', color: '#b6ff5c' },
  { id: 'pelvis', name: 'Pelvis', color: '#b98cff' }
];

/** @type {Array<{id: string, name: string, group: string, node?: boolean, frame?: boolean}>} */
export const STRUCTURES = [
  // Head & neck
  { id: 'brain', name: 'Brain', group: 'hn' },
  { id: 'brainstem', name: 'Brainstem', group: 'hn' },
  { id: 'eyes', name: 'Eyes & optic nerves', group: 'hn' },
  { id: 'parotids', name: 'Parotid glands', group: 'hn' },
  { id: 'submandibular', name: 'Submandibular glands', group: 'hn' },
  { id: 'oral', name: 'Oral cavity', group: 'hn' },
  { id: 'larynx', name: 'Larynx', group: 'hn' },
  { id: 'thyroid', name: 'Thyroid', group: 'hn' },
  { id: 'cord', name: 'Spinal cord', group: 'hn', frame: false },
  { id: 'nodes-hn', name: 'Cervical lymph nodes (I–V)', group: 'hn', node: true },
  // Chest
  { id: 'lungs', name: 'Lungs', group: 'chest' },
  { id: 'airways', name: 'Trachea & bronchi', group: 'chest' },
  { id: 'esophagus', name: 'Esophagus', group: 'chest', frame: false },
  { id: 'nodes-chest', name: 'Axillary, supraclavicular & IMN nodes', group: 'chest', node: true },
  // Heart & great vessels
  { id: 'lv', name: 'Left ventricle', group: 'heart' },
  { id: 'rv', name: 'Right ventricle', group: 'heart' },
  { id: 'la', name: 'Left atrium', group: 'heart' },
  { id: 'ra', name: 'Right atrium', group: 'heart' },
  { id: 'coronary', name: 'Coronary arteries', group: 'heart' },
  { id: 'aorta', name: 'Aorta', group: 'heart', frame: false },
  { id: 'arch-branches', name: 'Supra-aortic arteries', group: 'heart', frame: false },
  { id: 'pa', name: 'Pulmonary arteries', group: 'heart' },
  { id: 'pv', name: 'Pulmonary veins', group: 'heart' },
  { id: 'svc', name: 'Superior vena cava', group: 'heart' },
  { id: 'ivc', name: 'Inferior vena cava', group: 'heart', frame: false },
  { id: 'nodes-mediastinum', name: 'Mediastinal & hilar nodes', group: 'heart', node: true },
  // Abdomen
  { id: 'liver', name: 'Liver', group: 'abdomen' },
  { id: 'gallbladder', name: 'Gallbladder', group: 'abdomen' },
  { id: 'stomach', name: 'Stomach', group: 'abdomen' },
  { id: 'spleen', name: 'Spleen', group: 'abdomen' },
  { id: 'pancreas', name: 'Pancreas', group: 'abdomen' },
  { id: 'kidneys', name: 'Kidneys', group: 'abdomen' },
  { id: 'duodenum', name: 'Duodenum', group: 'abdomen' },
  { id: 'bowel', name: 'Small bowel', group: 'abdomen' },
  { id: 'colon', name: 'Colon', group: 'abdomen' },
  { id: 'nodes-paraaortic', name: 'Para-aortic nodes', group: 'abdomen', node: true },
  // Pelvis
  { id: 'bladder', name: 'Bladder', group: 'pelvis' },
  { id: 'rectum', name: 'Rectum & sigmoid', group: 'pelvis' },
  { id: 'femoral-heads', name: 'Femoral heads', group: 'pelvis' },
  { id: 'iliac-vessels', name: 'Iliac vessels', group: 'pelvis' },
  { id: 'prostate', name: 'Prostate', group: 'pelvis' },
  { id: 'seminal', name: 'Seminal vesicles', group: 'pelvis' },
  { id: 'nodes-pelvis', name: 'Iliac, obturator & inguinal nodes', group: 'pelvis', node: true }
];
