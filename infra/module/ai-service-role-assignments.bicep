// Assigns the necessary roles to the AI project


param storageAccountName string
param principalId string


resource storageAccount 'Microsoft.Storage/storageAccounts@2022-09-01' existing = {
  name: storageAccountName
  scope: resourceGroup()
}

resource storageDataBlobReader 'Microsoft.Authorization/roleDefinitions@2022-04-01' existing = {
  name: '2a2b9908-6ea1-4ae2-8e65-a410df84e7d1'
  scope: resourceGroup()
}

resource storageDataBlobReaderAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  scope: storageAccount
  name: guid(storageAccount.id, storageDataBlobReader.id, principalId)
  properties: {
    principalId: principalId
    roleDefinitionId: storageDataBlobReader.id
    principalType: 'User' 
  }
}

resource storageBlobDataContributor 'Microsoft.Authorization/roleDefinitions@2022-04-01' existing = {
  name: 'ba92f5b4-2d11-453d-a403-e96b0029c9fe'
  scope: resourceGroup()
}

resource storageBlobDataContributorAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  scope: storageAccount
  name: guid(storageAccount.id, storageBlobDataContributor.id, principalId)
  properties: {
    principalId: principalId
    roleDefinitionId: storageBlobDataContributor.id
    principalType: 'User' 
  }
}
