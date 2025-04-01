envFilePath=".env"

if [ -f "$envFilePath" ]; then
    rm -f "$envFilePath"
fi
touch "$envFilePath"

echo "AZURE_LOCATION=$(azd env get-value location)" >> "$envFilePath"
echo "AZURE_TENANT_ID=$(azd env get-value 'tenant().tenantId')" >> "$envFilePath"
echo "RESOURCE_GROUP=$(azd env get-value resourceGroupName)" >> "$envFilePath"
echo "SUBSCRIPTION_ID=$(azd env get-value 'subscription().subscriptionId')" >> "$envFilePath"
echo "WORKSPACE_NAME=$(azd env get-value '${workspacename}${resourceToken}')" >> "$envFilePath"
