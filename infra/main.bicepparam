using './main.bicep'

// You can use readEnvironmentVariable to load values from your environment

param environmentName = readEnvironmentVariable('AZURE_ENV_NAME', 'environmentName')
param principalId = readEnvironmentVariable('AZURE_PRINCIPAL_ID', 'principalId')
