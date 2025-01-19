
targetScope = 'subscription'

@minLength(1)
@maxLength(64)
@description('Name of the the environment which is used to generate a short unique hash used in all resources.')
param environmentName string

@minLength(1)
@description('Primary location for all resources')
@allowed(['eastus', 'westus2'])
@metadata({
  azd: {
    type: 'location'
  }
})
param location string = 'eastus'

@description('Name of the resource group')
param resourceGroupName string = ''

@description('Name for the AI resource and used to derive name of dependent resources.')
param aiHubName string = 'hub-demo'

@description('Friendly name for your Hub resource')
param aiHubFriendlyName string = 'Agents Hub resource'

@description('Description of your Azure AI resource displayed in AI studio')
param aiHubDescription string = 'This is an example AI resource for use in Azure AI Studio.'

@description('Name for the AI project resources.')
param aiProjectName string = 'project-demo'


@description('Name of the Azure AI Services account')
param aiServicesName string = 'agentaiservices'


@description('The AI Service Account full ARM Resource ID. This is an optional field, and if not provided, the resource will be created.')
param aiServiceAccountResourceId string = ''

@description('The Ai Storage Account full ARM Resource ID. This is an optional field, and if not provided, the resource will be created.')
param aiStorageAccountResourceId string = ''

param timestamp string = utcNow()

// Variables
var abbrs = loadJsonContent('./abbreviations.json')
var resourceToken = toLower(uniqueString(subscription().id, environmentName, location, timestamp))
var tags = { 'azd-env-name': environmentName }
var name = toLower('${aiHubName}')

resource resourceGroup 'Microsoft.Resources/resourceGroups@2022-09-01' = {
  name: '${abbrs.resourcesResourceGroups}${environmentName}' 
  location: location
  tags: tags
}

// Dependent resources for the Azure Machine Learning workspace
module aiDependencies './agent/standard-dependent-resources.bicep' = {
  name: '${abbrs.cognitiveServicesAccounts}${resourceToken}'
  scope: resourceGroup
  params: {
    location: location
    storageName: 'st${resourceToken}'
    keyvaultName: 'kv${name}${resourceToken}'
    aiServicesName: '${aiServicesName}${resourceToken}'
    tags: tags


     aiServiceAccountResourceId: aiServiceAccountResourceId
     aiStorageAccountResourceId: aiStorageAccountResourceId
    }
}

module aiHub './agent/standard-ai-hub.bicep' = {
  name: '${abbrs.cognitiveServicesAIhub}${resourceToken}'
  scope: resourceGroup
  params: {
    // workspace organization
    aiHubName: '${name}${resourceToken}'
    aiHubFriendlyName: aiHubFriendlyName
    aiHubDescription: aiHubDescription
    location: location
    tags: tags

    aiServicesName: aiDependencies.outputs.aiServicesName

    keyVaultId: aiDependencies.outputs.keyvaultId
    storageAccountId: aiDependencies.outputs.storageId
  }
}


// App outputs
output AZURE_LOCATION string = location
output AZURE_TENANT_ID string = tenant().tenantId
output RESOURCE_GROUP string = resourceGroupName





