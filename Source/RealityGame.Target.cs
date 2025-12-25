using UnrealBuildTool;
using System.Collections.Generic;

public class RealityGameTarget : TargetRules
{
	public RealityGameTarget( TargetInfo Target) : base(Target)
	{
		Type = TargetType.Game;
		DefaultBuildSettings = BuildSettingsVersion.V4;
		IncludeOrderVersion = EngineIncludeOrderVersion.Unreal5_3;
		ExtraModuleNames.Add("RealityGame");
	}
}
