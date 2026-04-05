$Env:CONDA_EXE = "/csehome/m25csa014/Dlops_ass5/myconda/bin/conda"
$Env:_CONDA_EXE = "/csehome/m25csa014/Dlops_ass5/myconda/bin/conda"
$Env:_CE_M = $null
$Env:_CE_CONDA = $null
$Env:CONDA_PYTHON_EXE = "/csehome/m25csa014/Dlops_ass5/myconda/bin/python"
$Env:_CONDA_ROOT = "/csehome/m25csa014/Dlops_ass5/myconda"
$CondaModuleArgs = @{ChangePs1 = $True}

Import-Module "$Env:_CONDA_ROOT\shell\condabin\Conda.psm1" -ArgumentList $CondaModuleArgs

Remove-Variable CondaModuleArgs