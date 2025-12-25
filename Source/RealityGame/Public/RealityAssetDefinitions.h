#pragma once

#include "CoreMinimal.h"
#include "Engine/DataAsset.h"
#include "RealityTypes.h"
#include "RealityAssetDefinitions.generated.h"

// Defines a mapping between a conceptual asset (e.g. "Tree") and a specific asset path
USTRUCT(BlueprintType)
struct FRealityAssetMapping
{
    GENERATED_BODY()

    // The key used to look up this asset (e.g., "Tree", "Wall", "Floor")
    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    FName AssetKey;

    // The specific asset to load (Soft Reference to avoid hard memory dependencies)
    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    TSoftObjectPtr<UObject> Asset;
};

// A Data Asset that defines the "Visuals" for a specific Reality Archetype
UCLASS(BlueprintType)
class REALITYGAME_API URealityThemeDefinition : public UDataAsset
{
    GENERATED_BODY()

public:
    // The archetype this theme belongs to (e.g., Fantasy)
    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Theme")
    ERealityArchetype Archetype;

    // The list of asset mappings for this theme
    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Theme")
    TArray<FRealityAssetMapping> AssetMappings;
};
