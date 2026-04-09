import * as d3 from 'd3';

const NODES = [
    { id: 'MI', type: 'condition', label: 'Heart Attack' },
    { id: 'Stroke', type: 'condition', label: 'Stroke' },
    { id: 'PulmonaryEmbolism', type: 'condition', label: 'Pulmonary Embolism' },
    { id: 'Sepsis', type: 'condition', label: 'Sepsis' },
    { id: 'Anaphylaxis', type: 'condition', label: 'Anaphylaxis' },
    { id: 'Appendicitis', type: 'condition', label: 'Appendicitis' },
    { id: 'Meningitis', type: 'condition', label: 'Meningitis' },
    { id: 'DiabeticKetoacidosis', type: 'condition', label: 'Diabetic Ketoacidosis' },
    { id: 'ChestPain', type: 'symptom', label: 'Chest Pain' },
    { id: 'LeftArmPain', type: 'symptom', label: 'Left Arm Pain' },
    { id: 'Sweating', type: 'symptom', label: 'Sweating' },
    { id: 'Nausea', type: 'symptom', label: 'Nausea' },
    { id: 'Dyspnea', type: 'symptom', label: 'Dyspnea' },
    { id: 'FacialDrooping', type: 'symptom', label: 'Facial Drooping' },
    { id: 'ArmWeakness', type: 'symptom', label: 'Arm Weakness' },
    { id: 'SpeechDifficulty', type: 'symptom', label: 'Speech Difficulty' },
    { id: 'SuddenHeadache', type: 'symptom', label: 'Sudden Headache' },
    { id: 'VisionLoss', type: 'symptom', label: 'Vision Loss' },
    { id: 'SuddenDyspnea', type: 'symptom', label: 'Sudden Dyspnea' },
    { id: 'RapidHeartRate', type: 'symptom', label: 'Rapid Heart Rate' },
    { id: 'CoughingBlood', type: 'symptom', label: 'Coughing Blood' },
    { id: 'Fever', type: 'symptom', label: 'Fever' },
    { id: 'RapidBreathing', type: 'symptom', label: 'Rapid Breathing' },
    { id: 'ConfusionSymptom', type: 'symptom', label: 'Confusion' },
    { id: 'LowBloodPressure', type: 'symptom', label: 'Low BP' },
    { id: 'HighHeartRate', type: 'symptom', label: 'High Heart Rate' },
    { id: 'Hives', type: 'symptom', label: 'Hives' },
    { id: 'ThroatSwelling', type: 'symptom', label: 'Throat Swelling' },
    { id: 'RightLowerQuadrantPain', type: 'symptom', label: 'Right Lower Quadrant Pain' },
    { id: 'Vomiting', type: 'symptom', label: 'Vomiting' },
    { id: 'LossOfAppetite', type: 'symptom', label: 'Loss of Appetite' },
    { id: 'SevereHeadache', type: 'symptom', label: 'Severe Headache' },
    { id: 'NeckStiffness', type: 'symptom', label: 'Neck Stiffness' },
    { id: 'Photophobia', type: 'symptom', label: 'Photophobia' },
    { id: 'Polyuria', type: 'symptom', label: 'Polyuria' },
    { id: 'Polydipsia', type: 'symptom', label: 'Polydipsia' },
    { id: 'AbdominalPain', type: 'symptom', label: 'Abdominal Pain' },
    { id: 'FruityBreath', type: 'symptom', label: 'Fruity Breath' },
    { id: 'ECG', type: 'test', label: 'ECG' },
    { id: 'TroponinTest', type: 'test', label: 'Troponin' },
    { id: 'ChestXRay', type: 'test', label: 'Chest X-Ray' },
    { id: 'CTScan', type: 'test', label: 'CT Scan' },
    { id: 'MRI', type: 'test', label: 'MRI' },
    { id: 'BloodTests', type: 'test', label: 'Blood Tests' },
    { id: 'CTAngiography', type: 'test', label: 'CT Angiography' },
    { id: 'DdimerTest', type: 'test', label: 'D-dimer Test' },
    { id: 'EchocardiogramTest', type: 'test', label: 'Echocardiogram' },
    { id: 'BloodCulture', type: 'test', label: 'Blood Culture' },
    { id: 'LactateTest', type: 'test', label: 'Lactate Test' },
    { id: 'CBC', type: 'test', label: 'CBC' },
    { id: 'UrineAnalysis', type: 'test', label: 'Urine Analysis' },
    { id: 'ClinicalDiagnosis', type: 'test', label: 'Clinical Diagnosis' },
    { id: 'TryptaseTest', type: 'test', label: 'Tryptase' },
    { id: 'UltrasoundTest', type: 'test', label: 'Ultrasound' },
    { id: 'LumbarPuncture', type: 'test', label: 'Lumbar Puncture' },
    { id: 'BloodGlucoseTest', type: 'test', label: 'Blood Glucose' },
    { id: 'UrineKetones', type: 'test', label: 'Urine Ketones' },
    { id: 'BloodGasTest', type: 'test', label: 'Blood Gas' },
    { id: 'Aspirin', type: 'treatment', label: 'Aspirin' },
    { id: 'Thrombolysis', type: 'treatment', label: 'Thrombolysis' },
    { id: 'PCI', type: 'treatment', label: 'Percutaneous Coronary Intervention' },
    { id: 'tPA', type: 'treatment', label: 'tPA' },
    { id: 'Thrombectomy', type: 'treatment', label: 'Thrombectomy' },
    { id: 'Anticoagulants', type: 'treatment', label: 'Anticoagulants' },
    { id: 'Heparin', type: 'treatment', label: 'Heparin' },
    { id: 'Warfarin', type: 'treatment', label: 'Warfarin' },
    { id: 'IVAntibiotics', type: 'treatment', label: 'IV Antibiotics' },
    { id: 'IVFluids', type: 'treatment', label: 'IV Fluids' },
    { id: 'Vasopressors', type: 'treatment', label: 'Vasopressors' },
    { id: 'Epinephrine', type: 'treatment', label: 'Epinephrine' },
    { id: 'Antihistamines', type: 'treatment', label: 'Antihistamines' },
    { id: 'Corticosteroids', type: 'treatment', label: 'Corticosteroids' },
    { id: 'Appendectomy', type: 'treatment', label: 'Appendectomy' },
    { id: 'SupportiveCare', type: 'treatment', label: 'Supportive Care' },
    { id: 'InsulinTherapy', type: 'treatment', label: 'Insulin Therapy' },
    { id: 'ElectrolyteReplacement', type: 'treatment', label: 'Electrolyte Replacement' },
    { id: 'Hypertension', type: 'risk', label: 'Hypertension' },
    { id: 'Smoking', type: 'risk', label: 'Smoking' },
    { id: 'Diabetes', type: 'risk', label: 'Diabetes' },
    { id: 'Obesity', type: 'risk', label: 'Obesity' },
    { id: 'AtrialFibrillation', type: 'risk', label: 'Atrial Fibrillation' },
    { id: 'Immobility', type: 'risk', label: 'Immobility' },
    { id: 'DeepVeinThrombosis', type: 'risk', label: 'Deep Vein Thrombosis' },
    { id: 'Surgery', type: 'risk', label: 'Surgery' },
    { id: 'Pregnancy', type: 'risk', label: 'Pregnancy' },
    { id: 'ImmunocompromisedState', type: 'risk', label: 'Immunocompromised State' },
    { id: 'RecentInfection', type: 'risk', label: 'Recent Infection' },
    { id: 'AgeFactor', type: 'risk', label: 'Age' },
    { id: 'ChronicIllness', type: 'risk', label: 'Chronic Illness' },
    { id: 'FoodAllergy', type: 'risk', label: 'Food Allergy' },
    { id: 'DrugAllergy', type: 'risk', label: 'Drug Allergy' },
    { id: 'InsectVenom', type: 'risk', label: 'Insect Venom' },
    { id: 'PreviousAnaphylaxis', type: 'risk', label: 'Prior Anaphylaxis' },
    { id: 'FamilyHistory', type: 'risk', label: 'Family History' },
    { id: 'GenderFactor', type: 'risk', label: 'Gender' },
    { id: 'YoungAge', type: 'risk', label: 'Young Age' },
    { id: 'Crowding', type: 'risk', label: 'Crowding' },
    { id: 'UnvaccinatedStatus', type: 'risk', label: 'Unvaccinated' },
    { id: 'Type1Diabetes', type: 'risk', label: 'Type 1 Diabetes' },
    { id: 'MissedInsulin', type: 'risk', label: 'Missed Insulin' },
    { id: 'Infection', type: 'risk', label: 'Infection' },
    { id: 'Stress', type: 'risk', label: 'Stress' },
];

const LINKS = [
    { source: 'MI', target: 'ChestPain', rel: 'symptom' },
    { source: 'MI', target: 'LeftArmPain', rel: 'symptom' },
    { source: 'MI', target: 'Sweating', rel: 'symptom' },
    { source: 'MI', target: 'Nausea', rel: 'symptom' },
    { source: 'MI', target: 'Dyspnea', rel: 'symptom' },
    { source: 'MI', target: 'ECG', rel: 'test' },
    { source: 'MI', target: 'TroponinTest', rel: 'test' },
    { source: 'MI', target: 'ChestXRay', rel: 'test' },
    { source: 'MI', target: 'Aspirin', rel: 'treatment' },
    { source: 'MI', target: 'Thrombolysis', rel: 'treatment' },
    { source: 'MI', target: 'PCI', rel: 'treatment' },
    { source: 'MI', target: 'Hypertension', rel: 'risk' },
    { source: 'MI', target: 'Smoking', rel: 'risk' },
    { source: 'MI', target: 'Diabetes', rel: 'risk' },
    { source: 'MI', target: 'Obesity', rel: 'risk' },
    { source: 'Stroke', target: 'FacialDrooping', rel: 'symptom' },
    { source: 'Stroke', target: 'ArmWeakness', rel: 'symptom' },
    { source: 'Stroke', target: 'SpeechDifficulty', rel: 'symptom' },
    { source: 'Stroke', target: 'SuddenHeadache', rel: 'symptom' },
    { source: 'Stroke', target: 'VisionLoss', rel: 'symptom' },
    { source: 'Stroke', target: 'CTScan', rel: 'test' },
    { source: 'Stroke', target: 'MRI', rel: 'test' },
    { source: 'Stroke', target: 'BloodTests', rel: 'test' },
    { source: 'Stroke', target: 'tPA', rel: 'treatment' },
    { source: 'Stroke', target: 'Thrombectomy', rel: 'treatment' },
    { source: 'Stroke', target: 'Anticoagulants', rel: 'treatment' },
    { source: 'Stroke', target: 'Hypertension', rel: 'risk' },
    { source: 'Stroke', target: 'Smoking', rel: 'risk' },
    { source: 'Stroke', target: 'AtrialFibrillation', rel: 'risk' },
    { source: 'Stroke', target: 'Diabetes', rel: 'risk' },
    { source: 'PulmonaryEmbolism', target: 'SuddenDyspnea', rel: 'symptom' },
    { source: 'PulmonaryEmbolism', target: 'ChestPain', rel: 'symptom' },
    { source: 'PulmonaryEmbolism', target: 'RapidHeartRate', rel: 'symptom' },
    { source: 'PulmonaryEmbolism', target: 'CoughingBlood', rel: 'symptom' },
    { source: 'PulmonaryEmbolism', target: 'CTAngiography', rel: 'test' },
    { source: 'PulmonaryEmbolism', target: 'DdimerTest', rel: 'test' },
    { source: 'PulmonaryEmbolism', target: 'EchocardiogramTest', rel: 'test' },
    { source: 'PulmonaryEmbolism', target: 'Heparin', rel: 'treatment' },
    { source: 'PulmonaryEmbolism', target: 'Warfarin', rel: 'treatment' },
    { source: 'PulmonaryEmbolism', target: 'Thrombolysis', rel: 'treatment' },
    { source: 'PulmonaryEmbolism', target: 'Immobility', rel: 'risk' },
    { source: 'PulmonaryEmbolism', target: 'DeepVeinThrombosis', rel: 'risk' },
    { source: 'PulmonaryEmbolism', target: 'Surgery', rel: 'risk' },
    { source: 'PulmonaryEmbolism', target: 'Pregnancy', rel: 'risk' },
    { source: 'Sepsis', target: 'Fever', rel: 'symptom' },
    { source: 'Sepsis', target: 'RapidBreathing', rel: 'symptom' },
    { source: 'Sepsis', target: 'ConfusionSymptom', rel: 'symptom' },
    { source: 'Sepsis', target: 'LowBloodPressure', rel: 'symptom' },
    { source: 'Sepsis', target: 'HighHeartRate', rel: 'symptom' },
    { source: 'Sepsis', target: 'BloodCulture', rel: 'test' },
    { source: 'Sepsis', target: 'LactateTest', rel: 'test' },
    { source: 'Sepsis', target: 'CBC', rel: 'test' },
    { source: 'Sepsis', target: 'UrineAnalysis', rel: 'test' },
    { source: 'Sepsis', target: 'IVAntibiotics', rel: 'treatment' },
    { source: 'Sepsis', target: 'IVFluids', rel: 'treatment' },
    { source: 'Sepsis', target: 'Vasopressors', rel: 'treatment' },
    { source: 'Sepsis', target: 'ImmunocompromisedState', rel: 'risk' },
    { source: 'Sepsis', target: 'RecentInfection', rel: 'risk' },
    { source: 'Sepsis', target: 'AgeFactor', rel: 'risk' },
    { source: 'Sepsis', target: 'ChronicIllness', rel: 'risk' },
    { source: 'Anaphylaxis', target: 'Hives', rel: 'symptom' },
    { source: 'Anaphylaxis', target: 'ThroatSwelling', rel: 'symptom' },
    { source: 'Anaphylaxis', target: 'Dyspnea', rel: 'symptom' },
    { source: 'Anaphylaxis', target: 'RapidHeartRate', rel: 'symptom' },
    { source: 'Anaphylaxis', target: 'LowBloodPressure', rel: 'symptom' },
    { source: 'Anaphylaxis', target: 'ClinicalDiagnosis', rel: 'test' },
    { source: 'Anaphylaxis', target: 'TryptaseTest', rel: 'test' },
    { source: 'Anaphylaxis', target: 'Epinephrine', rel: 'treatment' },
    { source: 'Anaphylaxis', target: 'Antihistamines', rel: 'treatment' },
    { source: 'Anaphylaxis', target: 'Corticosteroids', rel: 'treatment' },
    { source: 'Anaphylaxis', target: 'FoodAllergy', rel: 'risk' },
    { source: 'Anaphylaxis', target: 'DrugAllergy', rel: 'risk' },
    { source: 'Anaphylaxis', target: 'InsectVenom', rel: 'risk' },
    { source: 'Anaphylaxis', target: 'PreviousAnaphylaxis', rel: 'risk' },
    { source: 'Appendicitis', target: 'RightLowerQuadrantPain', rel: 'symptom' },
    { source: 'Appendicitis', target: 'Nausea', rel: 'symptom' },
    { source: 'Appendicitis', target: 'Fever', rel: 'symptom' },
    { source: 'Appendicitis', target: 'Vomiting', rel: 'symptom' },
    { source: 'Appendicitis', target: 'LossOfAppetite', rel: 'symptom' },
    { source: 'Appendicitis', target: 'UltrasoundTest', rel: 'test' },
    { source: 'Appendicitis', target: 'CTScan', rel: 'test' },
    { source: 'Appendicitis', target: 'CBC', rel: 'test' },
    { source: 'Appendicitis', target: 'Appendectomy', rel: 'treatment' },
    { source: 'Appendicitis', target: 'IVAntibiotics', rel: 'treatment' },
    { source: 'Appendicitis', target: 'AgeFactor', rel: 'risk' },
    { source: 'Appendicitis', target: 'FamilyHistory', rel: 'risk' },
    { source: 'Appendicitis', target: 'GenderFactor', rel: 'risk' },
    { source: 'Meningitis', target: 'SevereHeadache', rel: 'symptom' },
    { source: 'Meningitis', target: 'NeckStiffness', rel: 'symptom' },
    { source: 'Meningitis', target: 'Fever', rel: 'symptom' },
    { source: 'Meningitis', target: 'Photophobia', rel: 'symptom' },
    { source: 'Meningitis', target: 'Vomiting', rel: 'symptom' },
    { source: 'Meningitis', target: 'LumbarPuncture', rel: 'test' },
    { source: 'Meningitis', target: 'BloodCulture', rel: 'test' },
    { source: 'Meningitis', target: 'CTScan', rel: 'test' },
    { source: 'Meningitis', target: 'IVAntibiotics', rel: 'treatment' },
    { source: 'Meningitis', target: 'Corticosteroids', rel: 'treatment' },
    { source: 'Meningitis', target: 'SupportiveCare', rel: 'treatment' },
    { source: 'Meningitis', target: 'YoungAge', rel: 'risk' },
    { source: 'Meningitis', target: 'ImmunocompromisedState', rel: 'risk' },
    { source: 'Meningitis', target: 'Crowding', rel: 'risk' },
    { source: 'Meningitis', target: 'UnvaccinatedStatus', rel: 'risk' },
    { source: 'DiabeticKetoacidosis', target: 'Polyuria', rel: 'symptom' },
    { source: 'DiabeticKetoacidosis', target: 'Polydipsia', rel: 'symptom' },
    { source: 'DiabeticKetoacidosis', target: 'Vomiting', rel: 'symptom' },
    { source: 'DiabeticKetoacidosis', target: 'AbdominalPain', rel: 'symptom' },
    { source: 'DiabeticKetoacidosis', target: 'FruityBreath', rel: 'symptom' },
    { source: 'DiabeticKetoacidosis', target: 'BloodGlucoseTest', rel: 'test' },
    { source: 'DiabeticKetoacidosis', target: 'UrineKetones', rel: 'test' },
    { source: 'DiabeticKetoacidosis', target: 'BloodGasTest', rel: 'test' },
    { source: 'DiabeticKetoacidosis', target: 'InsulinTherapy', rel: 'treatment' },
    { source: 'DiabeticKetoacidosis', target: 'IVFluids', rel: 'treatment' },
    { source: 'DiabeticKetoacidosis', target: 'ElectrolyteReplacement', rel: 'treatment' },
    { source: 'DiabeticKetoacidosis', target: 'Type1Diabetes', rel: 'risk' },
    { source: 'DiabeticKetoacidosis', target: 'MissedInsulin', rel: 'risk' },
    { source: 'DiabeticKetoacidosis', target: 'Infection', rel: 'risk' },
    { source: 'DiabeticKetoacidosis', target: 'Stress', rel: 'risk' },
];

// Warm light palette matching the site
const TYPE_COLOR = {
    condition: '#c0392b',
    symptom: '#c8841a',
    test: '#2d6a4f',
    treatment: '#1a6b8a',
    risk: '#7d6e8a',
};

const TYPE_LABEL = {
    condition: 'Condition',
    symptom: 'Symptom',
    test: 'Diagnostic Test',
    treatment: 'Treatment',
    risk: 'Risk Factor',
};

const LINK_COLOR = {
    symptom: '#e8b84b',
    test: '#52a88a',
    treatment: '#4a9fc0',
    risk: '#b5a4c8',
};

export function initGraph(containerId) {
    const container = document.getElementById(containerId);
    if (!container) return;

    // ── Toolbar ───────────────────────────────────────────────────────────────
    const toolbar = document.createElement('div');
    toolbar.className = 'graph-toolbar';
    toolbar.innerHTML = `
        <div class="graph-legend-items">
            ${Object.entries(TYPE_LABEL).map(([t, l]) =>
        `<button class="graph-pill active" data-type="${t}">
                    <span class="graph-pill-dot" style="background:${TYPE_COLOR[t]}"></span>${l}
                 </button>`
    ).join('')}
        </div>
        <button class="graph-reset" title="Reset view">
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/>
                <path d="M3 3v5h5"/>
            </svg>
        </button>
    `;
    container.appendChild(toolbar);

    // ── Info panel ────────────────────────────────────────────────────────────
    const infoPanel = document.createElement('div');
    infoPanel.className = 'graph-info hidden';
    container.appendChild(infoPanel);

    // ── Hint ──────────────────────────────────────────────────────────────────
    const hint = document.createElement('div');
    hint.className = 'graph-hint';
    hint.textContent = 'Click any node to explore its connections';
    container.appendChild(hint);

    // ── SVG ───────────────────────────────────────────────────────────────────
    function getSize() {
        return { W: container.clientWidth, H: container.clientHeight };
    }

    const svg = d3.select(container).append('svg')
        .attr('width', '100%').attr('height', '100%')
        .style('display', 'block');

    const defs = svg.append('defs');

    // Subtle dot-grid background pattern
    const pattern = defs.append('pattern')
        .attr('id', 'dotgrid').attr('x', 0).attr('y', 0)
        .attr('width', 32).attr('height', 32)
        .attr('patternUnits', 'userSpaceOnUse');
    pattern.append('circle').attr('cx', 1).attr('cy', 1).attr('r', 1)
        .attr('fill', '#c4bcb4').attr('opacity', 0.4);

    svg.append('rect').attr('width', '100%').attr('height', '100%')
        .attr('fill', '#faf8f5');
    svg.append('rect').attr('width', '100%').attr('height', '100%')
        .attr('fill', 'url(#dotgrid)');

    // Arrow markers
    Object.entries(LINK_COLOR).forEach(([rel, col]) => {
        defs.append('marker').attr('id', `arr-${rel}`)
            .attr('viewBox', '0 0 10 10').attr('refX', 22).attr('refY', 5)
            .attr('markerWidth', 4).attr('markerHeight', 4).attr('orient', 'auto-start-reverse')
            .append('path').attr('d', 'M1 1L9 5L1 9').attr('fill', 'none')
            .attr('stroke', col).attr('stroke-width', 2)
            .attr('stroke-linecap', 'round').attr('stroke-linejoin', 'round');
    });

    const g = svg.append('g');

    const zoom = d3.zoom().scaleExtent([0.15, 4])
        .on('zoom', e => g.attr('transform', e.transform));
    svg.call(zoom).on('dblclick.zoom', null);

    // ── State ─────────────────────────────────────────────────────────────────
    let activeFilters = new Set(['condition', 'symptom', 'test', 'treatment', 'risk']);
    let selectedId = null;

    let sim, linkEl, nodeEl, labelEl, nodeGroups;

    function render() {
        if (sim) sim.stop();
        g.selectAll('*').remove();
        const { W, H } = getSize();

        // Always simulate ALL nodes and links — filtering is visual only
        const nodes = NODES.map(n => ({ ...n }));
        const links = LINKS.map(l => ({ ...l }));

        sim = d3.forceSimulation(nodes)
            .force('link', d3.forceLink(links).id(d => d.id)
                .distance(d => d.rel === 'symptom' ? 100 : 88).strength(0.35))
            .force('charge', d3.forceManyBody()
                .strength(d => d.type === 'condition' ? -600 : -140))
            .force('center', d3.forceCenter(W / 2, H / 2).strength(0.08))
            .force('collision', d3.forceCollide()
                .radius(d => d.type === 'condition' ? 48 : 26));

        const linkGroup = g.append('g');
        const nodeGroup = g.append('g');

        linkEl = linkGroup.selectAll('line').data(links).enter().append('line')
            .attr('class', d => `graph-link rel-${d.rel}`)
            .attr('stroke', d => LINK_COLOR[d.rel] || '#c4bcb4')
            .attr('stroke-width', 1.2)
            .attr('stroke-opacity', 0.35)
            .attr('marker-end', d => `url(#arr-${d.rel})`);

        nodeGroups = nodeGroup.selectAll('g').data(nodes).enter().append('g')
            .attr('class', d => `graph-node ntype-${d.type}`)
            .style('cursor', 'pointer')
            .call(d3.drag()
                .on('start', (e, d) => { if (!e.active) sim.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
                .on('drag', (e, d) => { d.fx = e.x; d.fy = e.y; })
                .on('end', (e, d) => { if (!e.active) sim.alphaTarget(0); d.fx = null; d.fy = null; })
            )
            .on('click', (e, d) => {
                selectedId = selectedId === d.id ? null : d.id;
                updateHighlight();
                if (selectedId) showInfo(d); else hideInfo();
                hint.classList.toggle('hidden', !!selectedId);
                e.stopPropagation();
            });

        // Condition outer ring
        nodeGroups.filter(d => d.type === 'condition').append('circle')
            .attr('r', 30)
            .attr('fill', d => `${TYPE_COLOR[d.type]}08`)
            .attr('stroke', d => TYPE_COLOR[d.type])
            .attr('stroke-width', 1)
            .attr('stroke-opacity', 0.25)
            .attr('stroke-dasharray', '4 4');

        nodeEl = nodeGroups.append('circle')
            .attr('r', d => d.type === 'condition' ? 22 : 8)
            .attr('fill', d => d.type === 'condition'
                ? TYPE_COLOR[d.type]
                : `${TYPE_COLOR[d.type]}20`)
            .attr('stroke', d => TYPE_COLOR[d.type])
            .attr('stroke-width', d => d.type === 'condition' ? 0 : 1.2)
            .attr('stroke-opacity', 0.8);

        labelEl = nodeGroups.append('text')
            .text(d => d.label)
            .attr('text-anchor', 'middle')
            .attr('dy', d => d.type === 'condition' ? 36 : 18)
            .attr('fill', d => d.type === 'condition' ? '#2a2420' : '#6d5b4f')
            .attr('font-size', d => d.type === 'condition' ? '10px' : '8.5px')
            .attr('font-family', "'Source Sans 3', system-ui, sans-serif")
            .attr('font-weight', d => d.type === 'condition' ? '600' : '400')
            .attr('pointer-events', 'none');

        sim.on('tick', () => {
            linkEl.attr('x1', d => d.source.x).attr('y1', d => d.source.y)
                .attr('x2', d => d.target.x).attr('y2', d => d.target.y);
            nodeGroups.attr('transform', d => `translate(${d.x},${d.y})`);
        });

        svg.on('click', () => {
            selectedId = null;
            updateHighlight();
            hideInfo();
            hint.classList.remove('hidden');
        });

        applyFilter();
    }

    function applyFilter() {
        if (!linkEl || !nodeGroups) return;
        linkEl.attr('display', d => activeFilters.has(d.rel) ? null : 'none');
        nodeGroups.attr('display', d =>
            d.type === 'condition' || activeFilters.has(d.type) ? null : 'none'
        );
        updateHighlight();
    }

    function updateHighlight() {
        if (!linkEl || !nodeGroups) return;
        if (!selectedId) {
            linkEl.attr('stroke-opacity', d => activeFilters.has(d.rel) ? 0.35 : 0)
                .attr('stroke-width', 1.2);
            nodeGroups.attr('opacity', 1);
            return;
        }
        const conn = new Set([selectedId]);
        linkEl.each(d => {
            const s = d.source.id || d.source, t = d.target.id || d.target;
            if ((s === selectedId || t === selectedId) && activeFilters.has(d.rel)) {
                conn.add(s); conn.add(t);
            }
        });
        linkEl
            .attr('stroke-opacity', d => {
                if (!activeFilters.has(d.rel)) return 0;
                const s = d.source.id || d.source, t = d.target.id || d.target;
                return (s === selectedId || t === selectedId) ? 0.8 : 0.03;
            })
            .attr('stroke-width', d => {
                const s = d.source.id || d.source, t = d.target.id || d.target;
                return (s === selectedId || t === selectedId) ? 2 : 0.5;
            });
        nodeGroups.attr('opacity', d => conn.has(d.id) ? 1 : 0.08);
    }

    function showInfo(d) {
        if (d.type === 'condition') {
            const byRel = { symptom: [], test: [], treatment: [], risk: [] };
            LINKS.forEach(l => {
                const s = l.source.id || l.source, t = l.target.id || l.target;
                if (s === d.id || t === d.id) {
                    const other = s === d.id ? t : s;
                    const node = NODES.find(n => n.id === other);
                    if (node && byRel[l.rel]) byRel[l.rel].push(node.label);
                }
            });
            infoPanel.innerHTML = `
                <div class="info-title">${d.label}</div>
                ${Object.entries(byRel).filter(([, v]) => v.length).map(([rel, items]) => `
                    <div class="info-group">
                        <span class="info-group-label" style="color:${TYPE_COLOR[rel]}">${TYPE_LABEL[rel]}s</span>
                        <span class="info-group-items">${items.join(', ')}</span>
                    </div>
                `).join('')}
            `;
        } else {
            const conditions = [];
            LINKS.forEach(l => {
                const s = l.source.id || l.source, t = l.target.id || l.target;
                if (s === d.id || t === d.id) {
                    const otherId = s === d.id ? t : s;
                    const node = NODES.find(n => n.id === otherId);
                    if (node) conditions.push(node.label);
                }
            });
            infoPanel.innerHTML = `
                <div class="info-title">${d.label}</div>
                <div class="info-group">
                    <span class="info-group-label" style="color:${TYPE_COLOR[d.type]}">${TYPE_LABEL[d.type]}</span>
                </div>
                <div class="info-group">
                    <span class="info-group-label" style="color:${TYPE_COLOR['condition']}">Associated Conditions</span>
                    <span class="info-group-items">${conditions.join(', ')}</span>
                </div>
            `;
        }

        infoPanel.classList.remove('hidden');
        attachSheetDrag(); // ← wire up swipe-to-dismiss every time panel opens
    }

    function hideInfo() {
        infoPanel.style.transform = '';
        infoPanel.classList.add('hidden');
    }

    // ── Mobile swipe-to-dismiss bottom sheet ──────────────────────────────────
    function attachSheetDrag() {
        if (window.innerWidth > 720) return;

        // Reuse existing handle if already injected, otherwise create it.
        // The real div replaces the CSS ::before pill so events can fire on it.
        let handle = infoPanel.querySelector('.sheet-handle');
        if (!handle) {
            handle = document.createElement('div');
            handle.className = 'sheet-handle';
            handle.style.cssText = [
                'position: relative',
                'width: 100%',
                'padding: 12px 0 12px',
                'display: flex',
                'align-items: center',
                'justify-content: center',
                'cursor: grab',
                'touch-action: none',   // prevents iOS pull-to-refresh on this element
                'flex-shrink: 0',
            ].join(';');

            const pill = document.createElement('div');
            pill.style.cssText = 'width:36px;height:4px;background:var(--border);border-radius:2px;pointer-events:none;';
            handle.appendChild(pill);
            infoPanel.insertBefore(handle, infoPanel.firstChild);
        }

        // Always remove old listeners before re-attaching to avoid stacking
        if (handle._ts) handle.removeEventListener('touchstart', handle._ts);
        if (handle._tm) handle.removeEventListener('touchmove', handle._tm);
        if (handle._te) window.removeEventListener('touchend', handle._te);

        let startY = 0, currentDelta = 0, dragging = false;

        function onTouchStart(e) {
            startY = e.touches[0].clientY;
            currentDelta = 0;
            dragging = true;
            infoPanel.style.transition = 'none';
        }

        function onTouchMove(e) {
            if (!dragging) return;
            e.preventDefault(); // blocks page scroll & iOS pull-to-refresh
            currentDelta = Math.max(0, e.touches[0].clientY - startY);
            infoPanel.style.transform = `translateY(${currentDelta}px)`;
        }

        function onTouchEnd() {
            if (!dragging) return;
            dragging = false;
            infoPanel.style.transition = ''; // restore CSS transition

            if (currentDelta > 80) {
                // dragged far enough — dismiss
                selectedId = null;
                updateHighlight();
                hideInfo();
                hint.classList.remove('hidden');
            } else {
                // snap back to fully open
                infoPanel.style.transform = '';
            }
        }

        handle._ts = onTouchStart;
        handle._tm = onTouchMove;
        handle._te = onTouchEnd;

        handle.addEventListener('touchstart', onTouchStart, { passive: true });
        handle.addEventListener('touchmove', onTouchMove, { passive: false }); // passive:false needed for preventDefault
        window.addEventListener('touchend', onTouchEnd);
    }

    // ── Filter pills — visual only, no re-render ──────────────────────────────
    toolbar.querySelectorAll('.graph-pill[data-type]').forEach(btn => {
        btn.addEventListener('click', e => {
            e.stopPropagation();
            const t = btn.dataset.type;
            if (t === 'condition') return;
            if (activeFilters.has(t)) {
                if ([...activeFilters].filter(x => x !== 'condition').length <= 1) return;
                activeFilters.delete(t);
                btn.classList.remove('active');
            } else {
                activeFilters.add(t);
                btn.classList.add('active');
            }
            selectedId = null;
            hideInfo();
            hint.classList.remove('hidden');
            applyFilter();
        });
    });

    toolbar.querySelector('.graph-reset').addEventListener('click', e => {
        e.stopPropagation();
        svg.transition().duration(500).call(zoom.transform, d3.zoomIdentity);
    });

    render();

    const ro = new ResizeObserver(() => {
        const { W, H } = getSize();
        if (sim) { sim.force('center', d3.forceCenter(W / 2, H / 2)); sim.alpha(0.2).restart(); }
    });
    ro.observe(container);
}
