#include "RealityProjectorComponent.h"
#include "Net/UnrealNetwork.h"
#include "DrawDebugHelpers.h"
#include "Math/Vector.h"

// Sets default values for this component's properties
URealityProjectorComponent::URealityProjectorComponent()
{
	// Set this component to be initialized when the game starts, and to be ticked every frame.  You can turn these features
	// off to improve performance if you don't need them.
	PrimaryComponentTick.bCanEverTick = true;
    SetIsReplicatedByDefault(true);

    // Initialize default values
    RealitySignature.Fidelity = 100.0f;
    RealitySignature.InfluenceRadius = 1000.0f;
    RealitySignature.ActiveStyle.Archetype = ERealityArchetype::Void;
}


// Called when the game starts
void URealityProjectorComponent::BeginPlay()
{
	Super::BeginPlay();
}


// Called every frame
void URealityProjectorComponent::TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction)
{
	Super::TickComponent(DeltaTime, TickType, ThisTickFunction);

    // Visual debug of the influence sphere
    // Use DrawDebugSphere to visually draw the sphere of influence
    if (GetOwner())
    {
        DrawDebugSphere(
            GetWorld(),
            GetOwner()->GetActorLocation(),
            RealitySignature.InfluenceRadius,
            32,
            FColor::Cyan,
            false,
            -1.0f,
            0,
            2.0f
        );
    }
}

void URealityProjectorComponent::GetLifetimeReplicatedProps(TArray<FLifetimeProperty>& OutLifetimeProps) const
{
    Super::GetLifetimeReplicatedProps(OutLifetimeProps);

    DOREPLIFETIME(URealityProjectorComponent, RealitySignature);
}

float URealityProjectorComponent::GetBlendWeightAtLocation(FVector Location, URealityProjectorComponent* Rival)
{
    if (!Rival || !GetOwner() || !Rival->GetOwner())
    {
        return 1.0f; // Default to self if no rival or invalid setup
    }

    float DistA = FVector::Dist(Location, GetOwner()->GetActorLocation());
    float DistB = FVector::Dist(Location, Rival->GetOwner()->GetActorLocation());

    // Avoid division by zero
    DistA = FMath::Max(DistA, 1.0f);
    DistB = FMath::Max(DistB, 1.0f);

    float StrengthA = RealitySignature.Fidelity / DistA;
    float StrengthB = Rival->RealitySignature.Fidelity / DistB;

    if (StrengthA + StrengthB <= KINDA_SMALL_NUMBER)
    {
        return 0.5f; // Neutral ground if both are effectively zero
    }

    // Return relative strength of this component (0.0 to 1.0)
    // If StrengthA > StrengthB, this should be > 0.5?
    // The prompt asks for "BlendAlpha" in CalculateRealityAtPoint to be StrengthB / StrengthA if A > B.
    // Let's implement CalculateRealityAtPoint first as the source of truth, and derive this if needed.

    // However, the prompt specifically says:
    // "Result.BlendAlpha = StrengthB / StrengthA; // Determines how "ghostly" Bob's world looks"
    // This implies BlendAlpha is "how much of the OTHER world is visible".

    // For this specific function "GetBlendWeightAtLocation", it usually implies "Weight of THIS component".
    // I will return StrengthA / (StrengthA + StrengthB) for a normalized weight,
    // OR follow the specific logic in CalculateRealityAtPoint.

    // Let's stick to the prompt's logic in CalculateRealityAtPoint and return the result from there.

    FBlendResult Result = CalculateRealityAtPoint(Location, Rival);

    // If we are the dominant archetype, the blend alpha is the "bleed" of the other.
    // So our weight is 1.0 - BlendAlpha.
    if (Result.DominantArchetype == RealitySignature.ActiveStyle.Archetype)
    {
        return 1.0f - Result.BlendAlpha;
    }
    else
    {
        // If we are not dominant, our weight is the BlendAlpha (the bleed of us into them).
        return Result.BlendAlpha;
    }
}

FBlendResult URealityProjectorComponent::CalculateRealityAtPoint(FVector Point, URealityProjectorComponent* Rival)
{
    FBlendResult Result;
    Result.DominantArchetype = ERealityArchetype::Void;
    Result.BlendAlpha = 0.0f;
    Result.bIsConflict = false;

    if (!Rival || !GetOwner() || !Rival->GetOwner())
    {
        Result.DominantArchetype = RealitySignature.ActiveStyle.Archetype;
        Result.BlendAlpha = 0.0f;
        return Result;
    }

    float DistA = FVector::Dist(Point, GetOwner()->GetActorLocation());
    float DistB = FVector::Dist(Point, Rival->GetOwner()->GetActorLocation());

    // Calculate "Strength" based on Distance + Fidelity
    // Prevent division by zero
    DistA = FMath::Max(DistA, 1.0f);
    DistB = FMath::Max(DistB, 1.0f);

    float StrengthA = (RealitySignature.Fidelity / DistA);
    float StrengthB = (Rival->RealitySignature.Fidelity / DistB);

    if (StrengthA >= StrengthB) {
        Result.DominantArchetype = RealitySignature.ActiveStyle.Archetype;
        // Avoid division by zero if StrengthA is 0 (unlikely given Max dist and non-zero fidelity, but good practice)
        if (StrengthA > KINDA_SMALL_NUMBER)
        {
             Result.BlendAlpha = StrengthB / StrengthA; // Determines how "ghostly" Bob's world looks
        }
        else
        {
            Result.BlendAlpha = 0.0f;
        }
    } else {
        Result.DominantArchetype = Rival->RealitySignature.ActiveStyle.Archetype;
         if (StrengthB > KINDA_SMALL_NUMBER)
        {
            Result.BlendAlpha = StrengthA / StrengthB;
        }
        else
        {
            Result.BlendAlpha = 0.0f;
        }
    }

    // Determine conflict (simplistic check: if archetypes are different)
    if (RealitySignature.ActiveStyle.Archetype != Rival->RealitySignature.ActiveStyle.Archetype)
    {
        Result.bIsConflict = true;
    }

    return Result;
}
