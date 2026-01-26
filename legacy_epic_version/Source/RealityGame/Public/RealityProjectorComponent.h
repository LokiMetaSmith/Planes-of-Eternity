#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "RealityTypes.h"
#include "RealityProjectorComponent.generated.h"

UCLASS( ClassGroup=(Custom), meta=(BlueprintSpawnableComponent) )
class REALITYGAME_API URealityProjectorComponent : public UActorComponent
{
	GENERATED_BODY()

public:
	// Sets default values for this component's properties
	URealityProjectorComponent();

protected:
	// Called when the game starts
	virtual void BeginPlay() override;

public:
	// Called every frame
	virtual void TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction) override;

    // The signature of the reality this component projects
    UPROPERTY(Replicated, EditAnywhere, BlueprintReadWrite, Category = "Reality")
    FRealitySignature RealitySignature;

    // Calculates the blend weight at a specific location against a rival component
    UFUNCTION(BlueprintCallable, Category = "Reality")
    float GetBlendWeightAtLocation(FVector Location, URealityProjectorComponent* Rival);

    // Calculates the full reality result at a point against a rival
    UFUNCTION(BlueprintCallable, Category = "Reality")
    FBlendResult CalculateRealityAtPoint(FVector Point, URealityProjectorComponent* Rival);

    // Replication
    virtual void GetLifetimeReplicatedProps(TArray<FLifetimeProperty>& OutLifetimeProps) const override;
};
