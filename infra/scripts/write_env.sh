$resourceGroup = ""
envFilePath=".env"

if [ -f "$envFilePath" ]; then
    rm -f "$envFilePath"
fi
touch "$envFilePath"


echo "AZURE_RESOURCE_GROUP=$resourceGroup" >> "$envFilePath"
echo "AZURE_SUBSCRIPTION_ID=$(az account show --query id -o tsv)" >> "$envFilePath"
workspaceName=$(az ml workspace list --resource-group $resourceGroup)
echo "AZUREML_WORKSPACE_NAME=$workspaceName" >> "$envFilePath"
