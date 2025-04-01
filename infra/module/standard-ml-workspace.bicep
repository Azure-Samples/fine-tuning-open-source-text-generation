@description('Azure region of the deployment')
param location string

@description('Name of the Azure ML workspace')
param workspaceName string

@description('Workspace description')
param workspaceDescription string = 'Azure Machine Learning Workspace'

@description('Resource ID of the key vault resource for storing connection strings')
param keyVaultId string

@description('Resource ID of the storage account resource for storing experimentation outputs')
param storageAccountId string

@description('Resource ID of the application insights resource for monitoring and logging')
param applicationInsightId string

@description('Resource ID of the container registry resource for storing images')
param containerRegistryId string

resource workspace 'Microsoft.MachineLearningServices/workspaces@2022-10-01' = {
  identity: {
    type: 'SystemAssigned'
  }
  name: workspaceName
  location: location
  properties: {
    friendlyName: workspaceName
    storageAccount: storageAccountId
    keyVault: keyVaultId
    description: workspaceDescription
    applicationInsights: applicationInsightId
    containerRegistry: containerRegistryId

  }
}

output amlWorkspaceId string = workspace.id


