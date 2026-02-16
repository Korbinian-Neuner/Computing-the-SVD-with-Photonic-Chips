package main;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;

import utils.Operation;

//Logger class used to collect and organize Operation logs,
//grouped into sections and tracked by time steps.
public class Logger {
	
	// Stores finished log sections mapped to a section name
	private static HashMap<ArrayList<Operation>, String> logs = new HashMap<ArrayList<Operation>, String>();
	
	// Current working list of logged operations
	private static ArrayList<Operation> log = new ArrayList<>();
	
	// Controls whether each logged operation increments the global timestep
	private static boolean incTimeStep = true;
	
	// Global timestep counter
	private static int timeStep = 0;
	
	// Saved timestep checkpoint
	private static int checkpoint = 0;
	
	// Determines if logging is currently active
	private static boolean active = true;
	
	// Resets all logging data and internal state
	public static void reset() {
		logs = new HashMap<ArrayList<Operation>, String>();
		log = new ArrayList<>();
		incTimeStep = true;
		timeStep = 0;
		checkpoint = 0;
	}
	
	// Enables or disables timestep incrementing
	public static void setIncTimeStep(boolean incTimeStep) {
		Logger.incTimeStep = incTimeStep;
		if(incTimeStep) {
			timeStep++; // Otherwise the next non parallel operation has the same timestep tag
		}
	}
	
	// Returns whether timestep incrementing is enabled
	public static boolean getIncTimeStep() {
		return incTimeStep;
	}
	
	// Adds a new Operation to the current log if logging is active
	public static void logOperation(String type) {
		if(active) {
			Operation o = new Operation(type, timeStep);
			log.add(o);
			if(incTimeStep) {
				timeStep++;
			}
		}
	}
	
	// Enables or disables logging globally
	public static void setActive(boolean active) {
		Logger.active = active;
	}
	
	// Saves the current timestep as a checkpoint
	public static void setcheckpoint() {
		checkpoint = timeStep;
	}
	
	// Restores the timestep from the saved checkpoint
	public static void gotoCheckpoint() {
		timeStep = checkpoint;
	}
	
	// Ends the current section and starts a new one
	public static void nextSection(String previousName) {
		logs.put(log, previousName); // Save current section with its name
		log = new ArrayList<Operation>(); // Start a new empty section
	}
	
	// Returns all logs, adding the currently active section as "Last"
	public static HashMap<ArrayList<Operation>, String> getLogs() {
		logs.put(log, "Last");
		return logs;
	}
	
	// Prints a summary of all logged sections and operation counts
	public static void printAll() {
		logs.put(log, "Last");
		
		Iterator ite = logs.keySet().iterator();
		while(ite.hasNext()) {
			ArrayList<Operation> clog = (ArrayList<Operation>) ite.next();
			if(clog.size() == 0) continue; // Skip empty sections
			System.out.println("Section " + logs.get(clog) + ":");
			
			// Counts occurrences of each operation type
			HashMap<String, Integer> summedTotals = new HashMap<String, Integer>();
			int maxTimeStep = 0;
			int minTimeStep = Integer.MAX_VALUE;
			
			Iterator it = clog.iterator();
			while(it.hasNext()) {
				Operation o = (Operation) it.next();
				
				// Count occurrences of operation types
				if(summedTotals.containsKey(o.getType())) {
					summedTotals.put(o.getType(), summedTotals.get(o.getType()) + 1);
				} else {
					summedTotals.put(o.getType(), 1);
				}
				
				// Track min and max timesteps
				if(o.getTimeStep() > maxTimeStep) {
					maxTimeStep = o.getTimeStep();
				}
				if(o.getTimeStep() < minTimeStep) {
					minTimeStep = o.getTimeStep();
				}
			}
			
			// Print timestep range information, relict from testing. Is still useful
			System.out.println("Timesteps: " + (maxTimeStep - minTimeStep) + "  " + minTimeStep + " - " + maxTimeStep);
			
			// Print counts per operation type
			it = summedTotals.keySet().iterator();
			while(it.hasNext()) {
				String op = (String) it.next();
				System.out.println(op + ":   " + summedTotals.get(op));
			}
		}
	}
	
	// Creates a simplified version of logs where operations with the same timestep
	// are merged into a single representative operation
	//  ONLY WORKS IF YOU DONT TOUCH THE LISTS
	// ALSO NO MULTITHREADING
	public static HashMap<ArrayList<Operation>, String> getParallelizedLogs() {
		HashMap<ArrayList<Operation>, String> ret = new HashMap<ArrayList<Operation>, String>();
		logs.put(log, "Last");
		
		Iterator ite = logs.keySet().iterator();
		while(ite.hasNext()) {
			ArrayList<Operation> clog = (ArrayList<Operation>) ite.next();
			ArrayList<Operation> parallelized = new ArrayList<Operation>();
			
			// Group operations by timestep
			HashMap<Integer, ArrayList<Operation>> timestepSorted = new HashMap<Integer, ArrayList<Operation>>();
			
			Iterator it = clog.iterator();
			while(it.hasNext()) {
				Operation op = (Operation) it.next();
				int timeStep = op.getTimeStep();
				if(timestepSorted.containsKey(timeStep)) {
					ArrayList<Operation> ops = timestepSorted.get(timeStep);
					ops.add(op);
				} else {
					ArrayList<Operation> ops = new ArrayList<Operation>();
					ops.add(op);
					timestepSorted.put(timeStep, ops);
				}
			}
			
			// Reduce grouped operations to a single representative per timestep
			it = timestepSorted.keySet().iterator();
			while(it.hasNext()) {
				Integer timeStep = (Integer) it.next();
				ArrayList<Operation> ops = timestepSorted.get(timeStep);
				
				//Not perfect
				parallelized.add(ops.get(0)); // Use first operation as representative
				for(int i = 1; i < ops.size(); i++) {
					// Warn if parallel operations differ in type
					if(!ops.get(i).getType().equals(ops.get(0).getType())) {
						System.out.println("WARNING! MERGING DIFFERENT PARALLEL OPERATIONS INTO THE SAME ONE");
						System.out.println(ops.get(0).getType() + " =!= " + ops.get(i).getType());
					}
				}
			}
			
			// Store simplified section
			ret.put(parallelized, logs.get(clog));
		}
		
		return ret;
	}
}
