
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
param workspacename string = 'waf-ml'

@description('Name of the Azure AI Services account')
param aiServicesName string = 'moduleaiservices'

@description('The AI Service Account full ARM Resource ID. This is an optional field, and if not provided, the resource will be created.')
param aiServiceAccountResourceId string = ''

@description('The Ai Storage Account full ARM Resource ID. This is an optional field, and if not provided, the resource will be created.')
param aiStorageAccountResourceId string = ''

param timestamp string = utcNow()

@description('The principal ID of the user or service principal that will be assigned the role.')
param principalId string


// Variables
var abbrs = loadJsonContent('./abbreviations.json')
var resourceToken = toLower(uniqueString(subscription().id, environmentName, location, timestamp))
var tags = { 'azd-env-name': environmentName }
var name = toLower('${workspacename}')

resource resourceGroup 'Microsoft.Resources/resourceGroups@2022-09-01' = {
  name: '${abbrs.resourcesResourceGroups}${environmentName}' 
  location: location
  tags: tags
}

// Dependent resources for the Azure Machine Learning workspace
module aiDependencies './module/standard-dependent-resources.bicep' = {
  name: '${abbrs.machineLearningAccounts}${resourceToken}'
  scope: resourceGroup
  params: {
    location: location
    storageName: 'st${resourceToken}'
    keyvaultName: 'kv${name}${resourceToken}'
    aiServicesName: '${aiServicesName}${resourceToken}'
    resourceToken: resourceToken
    tags: tags
     aiServiceAccountResourceId: aiServiceAccountResourceId
     aiStorageAccountResourceId: aiStorageAccountResourceId
    }
}

module aiHub './module/standard-ml-workspace.bicep' = {
  name: '${abbrs.machineLearningServicesWorkspaces}${resourceToken}'
  scope: resourceGroup
  params: {
    workspaceName: '${workspacename}${resourceToken}'
    location: location
    // tags: tags
    keyVaultId: aiDependencies.outputs.keyvaultId
    storageAccountId: aiDependencies.outputs.storageId
    containerRegistryId: aiDependencies.outputs.containerRegistryId
    applicationInsightId: aiDependencies.outputs.applicationInsightsId
    
  }
}



module aiServiceRoleAssignments './module/ai-service-role-assignments.bicep' = {
  name: 'aiserviceroleassignments${workspacename}${resourceToken}deployment'
  scope: resourceGroup
  params: {
    storageAccountName: aiDependencies.outputs.storageAccountName
    principalId: principalId
  }
}

// App outputs
output AZURE_LOCATION string = location
output AZURE_TENANT_ID string = tenant().tenantId
output RESOURCE_GROUP string = resourceGroupName
output SUBSCRIPTION_ID string = subscription().subscriptionId
output WORKSPACE_NAME string = '${workspacename}${resourceToken}'





