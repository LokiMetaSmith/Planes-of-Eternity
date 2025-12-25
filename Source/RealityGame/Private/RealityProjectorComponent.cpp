#include "RealityProjectorComponent.h"
#include "Net/UnrealNetwork.h"
#include "DrawDebugHelpers.h"
#include "Math/Vector.h"
#include "GameFramework/Actor.h"

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

float URealityProjectorComponent::GetBlendWeightAtLocation(FVector Location, const TArray<URealityProjectorComponent*>& Projectors)
{
    // Re-implement simplified logic here to identify if we are the winner or the runner-up.
    // CalculateRealityAtPoint gives us the "Result", but not "Who" is the RunnerUp.

    TArray<URealityProjectorComponent*> AllProjectors = Projectors;
    if (!AllProjectors.Contains(this))
    {
        AllProjectors.Add(this);
    }

    if (AllProjectors.Num() == 0 || !GetOwner())
    {
        return 0.0f;
    }

    float MaxStrength = -1.0f;
    float RunnerUpStrength = -1.0f;
    URealityProjectorComponent* Winner = nullptr;
    URealityProjectorComponent* RunnerUp = nullptr;

    // Calculate strengths
    for (URealityProjectorComponent* Candidate : AllProjectors)
    {
        if (!Candidate || !Candidate->GetOwner()) continue;

        float Dist = FVector::Dist(Location, Candidate->GetOwner()->GetActorLocation());
        Dist = FMath::Max(Dist, 1.0f); // Avoid div/0

        float Strength = Candidate->RealitySignature.Fidelity / Dist;

        if (Strength > MaxStrength)
        {
            RunnerUpStrength = MaxStrength;
            RunnerUp = Winner;

            MaxStrength = Strength;
            Winner = Candidate;
        }
        else if (Strength > RunnerUpStrength)
        {
            RunnerUpStrength = Strength;
            RunnerUp = Candidate;
        }
    }

    if (Winner == this)
    {
        // If we are the winner, our weight is 1.0 minus the bleed from the runner up
        if (MaxStrength > KINDA_SMALL_NUMBER)
        {
             float BlendAlpha = (RunnerUpStrength > 0.0f) ? (RunnerUpStrength / MaxStrength) : 0.0f;
             return 1.0f - BlendAlpha;
        }
        return 1.0f;
    }
    else if (RunnerUp == this)
    {
        // If we are the runner up, our weight is the bleed factor
        if (MaxStrength > KINDA_SMALL_NUMBER)
        {
            return RunnerUpStrength / MaxStrength;
        }
        return 0.0f; // Should not happen if MaxStrength is 0 since we are RunnerUp
    }

    // If we are neither winner nor runner up, we have no influence
    return 0.0f;
}

FBlendResult URealityProjectorComponent::CalculateRealityAtPoint(FVector Point, const TArray<URealityProjectorComponent*>& Projectors)
{
    FBlendResult Result;
    Result.DominantArchetype = ERealityArchetype::Void;
    Result.BlendAlpha = 0.0f;
    Result.bIsConflict = false;

    // Include "Self" in the calculation?
    // The signature implies we pass a list of projectors.
    // Usually "Self" should be in that list or added to it.
    // I will assume the caller passes *all* relevant projectors including this one if appropriate,
    // OR I should add myself to the consideration list.
    // Let's create a local list that includes Self.

    TArray<URealityProjectorComponent*> AllProjectors = Projectors;
    if (!AllProjectors.Contains(this))
    {
        AllProjectors.Add(this);
    }

    if (AllProjectors.Num() == 0)
    {
        return Result;
    }

    float MaxStrength = -1.0f;
    float RunnerUpStrength = -1.0f;
    URealityProjectorComponent* Winner = nullptr;
    URealityProjectorComponent* RunnerUp = nullptr;

    for (URealityProjectorComponent* Candidate : AllProjectors)
    {
        if (!Candidate || !Candidate->GetOwner()) continue;

        float Dist = FVector::Dist(Point, Candidate->GetOwner()->GetActorLocation());
        Dist = FMath::Max(Dist, 1.0f); // Avoid div/0

        float Strength = Candidate->RealitySignature.Fidelity / Dist;

        if (Strength > MaxStrength)
        {
            // Demote current winner to runner up
            RunnerUpStrength = MaxStrength;
            RunnerUp = Winner;

            // New winner
            MaxStrength = Strength;
            Winner = Candidate;
        }
        else if (Strength > RunnerUpStrength)
        {
            // New runner up
            RunnerUpStrength = Strength;
            RunnerUp = Candidate;
        }
    }

    if (Winner)
    {
        Result.DominantArchetype = Winner->RealitySignature.ActiveStyle.Archetype;

        if (MaxStrength > KINDA_SMALL_NUMBER)
        {
            // Bleed is RunnerUp / Winner
            if (RunnerUpStrength > 0.0f)
            {
                Result.BlendAlpha = RunnerUpStrength / MaxStrength;
            }
            else
            {
                Result.BlendAlpha = 0.0f;
            }
        }
        else
        {
             Result.BlendAlpha = 0.0f;
        }

        // Conflict?
        if (RunnerUp && Winner->RealitySignature.ActiveStyle.Archetype != RunnerUp->RealitySignature.ActiveStyle.Archetype)
        {
            Result.bIsConflict = true;
        }
    }

    return Result;
}
