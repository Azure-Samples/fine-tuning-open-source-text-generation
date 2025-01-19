// Creates an Azure AI resource with proxied endpoints for the Azure AI services provider

@description('Azure region of the deployment')
param location string

@description('Tags to add to the resources')
param tags object

@description('AI hub name')
param aiHubName string

@description('AI hub display name')
param aiHubFriendlyName string = aiHubName

@description('AI hub description')
param aiHubDescription string

@description('Resource ID of the key vault resource for storing connection strings')
param keyVaultId string

@description('Resource ID of the storage account resource for storing experimentation outputs')
param storageAccountId string

@description('Name AI Services resource')
param aiServicesName string

var acsConnectionName = '${aiHubName}-connection-AISearch'

var aoaiConnection  = '${aiHubName}-connection-AIServices_aoai'


resource aiServices 'Microsoft.CognitiveServices/accounts@2023-05-01' existing = {
  name: aiServicesName
  scope: resourceGroup()
}


resource aiHub 'Microsoft.MachineLearningServices/workspaces@2024-07-01-preview' = {
  name: aiHubName
  location: location
  tags: tags
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    // organization
    friendlyName: aiHubFriendlyName
    description: aiHubDescription

    // dependent resources
    keyVault: keyVaultId
    storageAccount: storageAccountId
  }
  kind: 'hub'

}

output aiHubID string = aiHub.id
output aoaiConnectionName string = aoaiConnection
output acsConnectionName string = acsConnectionName
