
document.getElementById('function').addEventListener('change', function() {
    var userInputDiv = document.getElementById('userInputs');
    if (this.value === 'y*k' || this.value === 'y+k' || this.value === 'y-k' || this.value === 'y/k' || this.value === 'y^k'|| this.value === 'k-y'|| this.value === 'k/y') {
        userInputDiv.style.display = 'block';
    } else {
        userInputDiv.style.display = 'none';
    }
});
document.getElementById('transformConcentration').addEventListener('change', function() {
    var userXInputDiv = document.getElementById('userXInputs');
    if (this.value === 'changeX0' || this.value === 'multConstant' || this.value === 'divConstant') {
        userXInputDiv.style.display = 'block';
    } else {
        userXInputDiv.style.display = 'none';
    }
});

document.getElementById('transform').addEventListener('change', function() {
    var normalizationOptions = document.getElementById('normalizationOptions');
    if (this.value === 'normalize') {
        normalizationOptions.style.display = 'block';
    } else {
        normalizationOptions.style.display = 'none';
    }
    var transformOptions = document.getElementById('transformOptions');
    if (this.value === 'transform') {
        transformOptions.style.display = 'block';
    } else {
        transformOptions.style.display = 'none';
    }
    var transformConcentrationOptions = document.getElementById('transformConcentrationOptions');
    if (this.value === 'transform_concentrations') {
        transformConcentrationOptions.style.display = 'block';
    } else {
        transformConcentrationOptions.style.display = 'none';
    }
    var auc = document.getElementById('areaUnderCurveDisplay');
    if (this.value === 'area_under_curve') {
        auc.style.display = 'block';
    } else {
        auc.style.display = 'none';
    }
    var fracOptions = document.getElementById('fracOptions');
    if (this.value === 'fraction_of_total') {
        fracOptions.style.display = 'block';
    } else {
        fracOptions.style.display = 'none';
    }
    var pruneOptions = document.getElementById('pruneOptions');
    if (this.value === 'prune') {
        pruneOptions.style.display = 'block';
    } else {
        pruneOptions.style.display = 'none';
    }
    var transposeDisplay = document.getElementById('transposeDisplay');
    if (this.value === 'transpose') {
        transposeDisplay.style.display = 'block';
    } else {
        transposeDisplay.style.display = 'none';
    }
    
});
document.getElementById('testNominal').addEventListener('change', function() {
    var binomOptions = document.getElementById('binomOptions');
    if (this.value === 'exact_test_of_goodness_of_fit') {
        binomOptions.style.display = 'block';
    } else {
        binomOptions.style.display = 'none';
    }
    var chigoodDisplay = document.getElementById('chigoodDisplay');
    if (this.value === 'chi_square_test_of_goodness_of_fit') {
        chigoodDisplay.style.display = 'block';
    } else {
        chigoodDisplay.style.display = 'none';
    }
    if (this.value === 'chi_square_test_of_independence') {
        chiindDisplay.style.display = 'block';
    } else {
        chiindDisplay.style.display = 'none';
    }
    
});

