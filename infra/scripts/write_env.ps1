$resourceGroup = "KA-SAND-RG"
$envFilePath = ".env"

If (Test-Path $envFilePath) {
    Remove-Item $envFilePath -Force
}
New-Item -Path $envFilePath -ItemType File -Force | Out-Null

Add-Content -Path $envFilePath -Value ("AZURE_RESOURCE_GROUP=" + $resourceGroup)

Add-Content -Path $envFilePath -Value ("AZURE_SUBSCRIPTION_ID=" + (az account show --query id -o tsv))

$workspaceName = (az ml workspace list --resource-group $resourceGroup)
Add-Content -Path $envFilePath -Value ("AZUREML_WORKSPACE_NAME=" + $workspaceName)

