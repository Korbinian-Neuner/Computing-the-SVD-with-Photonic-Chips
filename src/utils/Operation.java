package utils;

public class Operation {
	
	private String type;
	private int timeStep;
	
	public Operation(String type, int timeStep) {
		this.type = type;
		this.timeStep = timeStep;
	}
	
	public String getType() {
		return type;
	}
	
	public int getTimeStep() {
		return timeStep;
	}
}
