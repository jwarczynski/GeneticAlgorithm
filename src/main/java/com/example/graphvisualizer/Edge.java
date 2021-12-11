package com.example.graphvisualizer;

import javafx.fxml.FXML;
import javafx.scene.paint.Color;
import javafx.scene.shape.Line;

public class Edge {



    private double startX;
    private double startY;
    private double endX;
    private double endY;
    @FXML
    private Line skin;
    @FXML
    private Color color;

    public Edge(double startX, double startY, double endX, double endY) {
        this.startX = startX;
        this.startY = startY;
        this.endX = endX;
        this.endY = endY;
        color = Color.BLACK;
        skin = new Line();
        skin.setStartX(startX);
        skin.setStartY(startY);
        skin.setEndX(endX);
        skin.setEndY(endY);
        skin.setStrokeWidth(2);
        skin.setFill(color);
    }

    public Line getSkin() {
        return skin;
    }

    public void activate(Vortex a, Vortex b){
        if(a.getX() == startX && a.getY() == startY && b.getX() == endX && b.getY() == endY){
            color = Color.RED;
            skin.setFill(color);
        }
    }

    public void deactivate(){
        color = Color.BLACK;
        skin.setFill(color);
    }
}
