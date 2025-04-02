$envFilePath = ".env"

If (Test-Path $envFilePath) {
    Remove-Item $envFilePath -Force
}
New-Item -Path $envFilePath -ItemType File -Force | Out-Null

# Added outputs
Add-Content -Path $envFilePath -Value ("AZURE_LOCATION=" + (azd env get-value location))
Add-Content -Path $envFilePath -Value ("AZURE_TENANT_ID=" + (azd env get-value tenantId))
Add-Content -Path $envFilePath -Value ("RESOURCE_GROUP=" + (azd env get-value resourceGroupName))
Add-Content -Path $envFilePath -Value ("SUBSCRIPTION_ID=" + (azd env get-value subscriptionId))
Add-Content -Path $envFilePath -Value ("WORKSPACE_NAME=" + '${workspacename}${resourceToken}')
