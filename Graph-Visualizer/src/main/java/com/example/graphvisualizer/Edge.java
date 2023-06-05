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
        skin.setStroke(color);
    }

    public Line getSkin() {
        return skin;
    }

    public void activate(Vortex a){
        if((a.getX() == startX && a.getY() == startY )|| (a.getX() == endX && a.getY() == endY)){
            skin.setOpacity(1);
            color = Color.RED;
            skin.setStroke(color);
        }
    }

    public void deactivate(){
        skin.setOpacity(1);
        color = Color.BLACK;
        skin.setStroke(color);
    }
}
