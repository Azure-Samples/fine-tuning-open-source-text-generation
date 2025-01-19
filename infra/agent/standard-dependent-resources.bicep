// Creates Azure dependent resources for Azure AI studio

@description('Azure region of the deployment')
param location string = resourceGroup().location

@description('Tags to add to the resources')
param tags object = {}

@description('AI services name')
param aiServicesName string

@description('The name of the Key Vault')
param keyvaultName string

@description('Name of the storage account')
param storageName string

@description('The AI Service Account full ARM Resource ID. This is an optional field, and if not provided, the resource will be created.')
param aiServiceAccountResourceId string

@description('The AI Storage Account full ARM Resource ID. This is an optional field, and if not provided, the resource will be created.')
param aiStorageAccountResourceId string 

var aiServiceExists = aiServiceAccountResourceId != ''
var aiStorageExists = aiStorageAccountResourceId != ''

resource keyVault 'Microsoft.KeyVault/vaults@2022-07-01' = {
  name: keyvaultName
  location: location
  tags: tags
  properties: {
    createMode: 'default'
    enabledForDeployment: false
    enabledForDiskEncryption: false
    enabledForTemplateDeployment: false
    enableSoftDelete: true
    enableRbacAuthorization: true
    enablePurgeProtection: true
    networkAcls: {
      bypass: 'AzureServices'
      defaultAction: 'Deny'
    }
    sku: {
      family: 'A'
      name: 'standard'
    }
    softDeleteRetentionInDays: 7
    tenantId: subscription().tenantId
  }
}


var aiServiceParts = split(aiServiceAccountResourceId, '/')

resource existingAIServiceAccount 'Microsoft.CognitiveServices/accounts@2023-05-01' existing = if (aiServiceExists) {
  name: aiServiceParts[8]
  scope: resourceGroup(aiServiceParts[2], aiServiceParts[4])
}

var aiStorageParts = split(aiStorageAccountResourceId, '/')

resource existingAIStorageAccount 'Microsoft.Storage/storageAccounts@2023-05-01' existing = if (aiStorageExists) {
  name: aiStorageParts[8]
  scope: resourceGroup(aiStorageParts[2], aiStorageParts[4])
}

param sku string = 'Standard_LRS'

resource storage 'Microsoft.Storage/storageAccounts@2022-05-01' = if(!aiStorageExists) {
  name: storageName
  location: location
  kind: 'StorageV2'
  sku: {
    name: sku
  }
  properties: {
    minimumTlsVersion: 'TLS1_2'
    allowBlobPublicAccess: false
    publicNetworkAccess: 'Enabled'
    networkAcls: {
      bypass: 'AzureServices'
      defaultAction: 'Allow'
      virtualNetworkRules: []
    }
    allowSharedKeyAccess: false
  }
}

output aiServicesName string =  aiServiceExists ? existingAIServiceAccount.name : aiServicesName
output aiServiceAccountResourceGroupName string = aiServiceExists ? aiServiceParts[4] : resourceGroup().name
output aiServiceAccountSubscriptionId string = aiServiceExists ? aiServiceParts[2] : subscription().subscriptionId 

output storageAccountName string = aiStorageExists ? existingAIStorageAccount.name :  storage.name
output storageId string =  aiStorageExists ? existingAIStorageAccount.id :  storage.id
output storageAccountResourceGroupName string = aiStorageExists ? aiStorageParts[4] : resourceGroup().name
output storageAccountSubscriptionId string = aiStorageExists ? aiStorageParts[2] : subscription().subscriptionId

output keyvaultId string = keyVault.id
