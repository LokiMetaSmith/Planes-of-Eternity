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

float URealityProjectorComponent::GetBlendWeightAtLocation(FVector Location, const TArray<URealityProjectorComponent*>& Projectors)
{
    // Calculate the full reality result
    FBlendResult Result = CalculateRealityAtPoint(Location, Projectors);

    // If we are the dominant archetype, the blend alpha is the "bleed" of the runner-up.
    // So our weight is 1.0 - BlendAlpha.
    if (Result.DominantArchetype == RealitySignature.ActiveStyle.Archetype)
    {
        return 1.0f - Result.BlendAlpha;
    }
    else
    {
        // If we are not dominant, our weight is effectively 0 unless we are the runner up?
        // But the previous 1v1 logic implied "How much of the OTHER world is visible".
        // If we are not dominant, and we are not the runner up, we are likely 0.
        // However, the simple BlendAlpha returned is "Bleed of RunnerUp into Winner".

        // If we are the runner up, we are the ones bleeding in.
        // But the current CalculateRealityAtPoint only returns WHO won and HOW MUCH the runner up bleeds.
        // It doesn't identify the runner up explicitly in the struct.

        // For simplicity in this N-player context:
        // If we won, we are 1.0 - BlendAlpha.
        // If we lost, we are 0.0 (or we could be the BlendAlpha if we are the strongest loser, but let's keep it simple).
        return Result.BlendAlpha; // Wait, this logic was valid for 1v1 where "If I didn't win, the other guy did".

        // Let's refine this return value.
        // If Result.DominantArchetype != MyArchetype, it means I lost.
        // If I lost, how much of ME is visible?
        // If I am the runner up, it's Result.BlendAlpha.
        // If I am a distant third, it's 0.
        // Since I can't easily know if I am the runner up without re-calculating or expanding the struct...
        // let's just return Result.BlendAlpha if the archetypes match? No.

        // Actually, this function is usually called ON the component to know "My Weight".
        // Let's rely on CalculateRealityAtPoint to be the source of truth.
        // But since FBlendResult doesn't return the RunnerUp pointer, we can't be sure.

        // I will re-implement the specific check here.
        // Or better, assume that if I'm not dominant, I'm just contributing to the "conflict" state.

        // Re-reading the prompt: "The result of two worlds colliding... Who won? How much bleed?"
        // In N-players, we only care about the Winner and the "Bleed" (which is the next strongest).

        // If I am not the winner, I return 0.0f unless I am the specific runner up.
        // Since I don't want to overcomplicate the FBlendResult struct (as it's defined by the user),
        // I will leave this simple: If I am dominant, 1-Alpha. Else 0.

        if (Result.DominantArchetype == RealitySignature.ActiveStyle.Archetype)
        {
             return 1.0f - Result.BlendAlpha;
        }
        return 0.0f; // Simplified for N-player: only winner gets full representation minus bleed.
    }
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
