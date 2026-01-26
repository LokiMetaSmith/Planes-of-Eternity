#pragma once

#include "CoreMinimal.h"
#include "RealityTypes.generated.h"

// 1. The Broad Categories (The "Genre")
UENUM(BlueprintType)
enum class ERealityArchetype : uint8
{
    Void        UMETA(DisplayName = "The Void"), // Default/Empty
    Fantasy     UMETA(DisplayName = "High Fantasy"),
    SciFi       UMETA(DisplayName = "Cyber Punk"),
    Horror      UMETA(DisplayName = "Eldritch Horror"),
    Toon        UMETA(DisplayName = "Toon Logic")
};

// 2. The Specific Flavor (The "Style")
// This controls the specific asset pack used (e.g., Gothic vs. Elven)
USTRUCT(BlueprintType)
struct FRealityStyle
{
    GENERATED_BODY()

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    ERealityArchetype Archetype; // e.g., Fantasy

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    FName SubTheme; // e.g., "Necromancer_Castle"

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    int32 Seed; // The deterministic random number for WFC generation
};

// Placeholder for Injection
USTRUCT(BlueprintType)
struct FRealityInjection
{
    GENERATED_BODY()

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    FName InjectionID;

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    TSoftClassPtr<class AActor> AssetClass;

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    FTransform RelativeTransform;
};

// 3. The Player State (The "Who")
// This is the data packet that every player carries around. This is what gets sent over the network.
USTRUCT(BlueprintType)
struct FRealitySignature
{
    GENERATED_BODY()

    // The visual look the player is projecting
    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    FRealityStyle ActiveStyle;

    // The "Power" of their computer/magic.
    // Higher number = Higher Poly Count, Better Physics, Harder to overwite.
    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    float Fidelity;

    // The radius of their influence bubble in world units (cm)
    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    float InfluenceRadius;

    // A list of "Injections" (e.g., The Sci-Fi Turret inside a Fantasy World)
    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    TArray<FRealityInjection> ActiveInjections;
};

// 4. The Resolver Logic Result
// The result of two worlds colliding at a specific point in space
USTRUCT(BlueprintType)
struct FBlendResult
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadOnly)
    ERealityArchetype DominantArchetype; // Who won?

    UPROPERTY(BlueprintReadOnly)
    float BlendAlpha; // 0.0 to 1.0 (How much "Bleed" is happening?)

    UPROPERTY(BlueprintReadOnly)
    bool bIsConflict; // True if genres are opposites (e.g., SciFi vs Fantasy)
};
