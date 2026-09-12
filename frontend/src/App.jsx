import { useState } from "react";
import {
  MapContainer,
  TileLayer,
  Polyline,
  Marker,
  Popup,
  useMap
} from "react-leaflet";

import L from "leaflet";
import "leaflet/dist/leaflet.css";
import "./App.css";


// ============================================================
// FIX DEFAULT LEAFLET MARKER ICON
// ============================================================

delete L.Icon.Default.prototype._getIconUrl;

L.Icon.Default.mergeOptions({
  iconRetinaUrl:
    "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon-2x.png",

  iconUrl:
    "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon.png",

  shadowUrl:
    "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-shadow.png"
});


// ============================================================
// SOURCE / DESTINATION
// ============================================================

const INITIAL_SOURCE = {
  lat: 26.1445,
  lon: 91.7362
};

const INITIAL_DESTINATION = {
  lat: 26.3509,
  lon: 92.6836
};


// ============================================================
// MAP FIT COMPONENT
// ============================================================

function FitRoute({ route }) {

  const map = useMap();

  if (
    route &&
    route.geometry &&
    route.geometry.coordinates
  ) {

    const coordinates =
      route.geometry.coordinates.map(
        ([lon, lat]) => [lat, lon]
      );

    if (coordinates.length > 0) {

      const bounds =
        L.latLngBounds(coordinates);

      map.fitBounds(
        bounds,
        {
          padding: [30, 30]
        }
      );
    }
  }

  return null;
}


// ============================================================
// RISK COLOR
// ============================================================

function getRiskColor(
  hazard
) {

  if (hazard >= 0.80) {
    return "#dc2626";
  }

  if (hazard >= 0.60) {
    return "#f97316";
  }

  if (hazard >= 0.40) {
    return "#eab308";
  }

  return "#16a34a";
}


// ============================================================
// CHECKPOINT ICON
// ============================================================

function createCheckpointIcon(
  number
) {

  return L.divIcon({

    className:
      "checkpoint-marker",

    html: `
      <div class="checkpoint-circle">
        ${number}
      </div>
    `,

    iconSize: [30, 30],

    iconAnchor: [15, 15]
  });
}


// ============================================================
// APP
// ============================================================

function App() {

  const [source, setSource] =
    useState(INITIAL_SOURCE);

  const [destination, setDestination] =
    useState(INITIAL_DESTINATION);

  const [route, setRoute] =
    useState(null);

  const [checkpoints, setCheckpoints] =
    useState([]);

  const [summary, setSummary] =
    useState(null);

  const [loading, setLoading] =
    useState(false);

  const [error, setError] =
    useState("");


  // ==========================================================
  // FETCH ROUTE
  // ==========================================================

  const findRoute = async () => {

    setLoading(true);
    setError("");

    try {

      const response =
        await fetch(
          "http://127.0.0.1:8000/route",
          {
            method: "POST",

            headers: {
              "Content-Type":
                "application/json"
            },

            body: JSON.stringify({
              source,
              destination
            })
          }
        );


      if (!response.ok) {

        const errorData =
          await response.json();

        throw new Error(
          errorData.detail ||
          "Route request failed"
        );
      }


      const data =
        await response.json();


      // Store API result
      setRoute(data.route);

      setSummary(data.summary);

      setCheckpoints(
        data.checkpoints || []
      );

    } catch (err) {

      console.error(err);

      setError(
        err.message ||
        "Something went wrong"
      );

    } finally {

      setLoading(false);
    }
  };


  // ==========================================================
  // RENDER
  // ==========================================================

  return (

    <div className="app">


      {/* ==================================================
          HEADER
          ================================================== */}

      <header className="header">

        <div>

          <h1>
            NER Smart Route Intelligence
          </h1>

          <p>
            AI-powered risk-aware logistics routing
          </p>

        </div>

      </header>


      {/* ==================================================
          CONTROL PANEL
          ================================================== */}

      <section className="control-panel">


        {/* SOURCE */}

        <div className="input-group">

          <label>
            Source Latitude
          </label>

          <input
            type="number"
            step="any"
            value={source.lat}
            onChange={(e) =>
              setSource({
                ...source,
                lat:
                  Number(e.target.value)
              })
            }
          />

        </div>


        <div className="input-group">

          <label>
            Source Longitude
          </label>

          <input
            type="number"
            step="any"
            value={source.lon}
            onChange={(e) =>
              setSource({
                ...source,
                lon:
                  Number(e.target.value)
              })
            }
          />

        </div>


        {/* DESTINATION */}

        <div className="input-group">

          <label>
            Destination Latitude
          </label>

          <input
            type="number"
            step="any"
            value={destination.lat}
            onChange={(e) =>
              setDestination({
                ...destination,
                lat:
                  Number(e.target.value)
              })
            }
          />

        </div>


        <div className="input-group">

          <label>
            Destination Longitude
          </label>

          <input
            type="number"
            step="any"
            value={destination.lon}
            onChange={(e) =>
              setDestination({
                ...destination,
                lon:
                  Number(e.target.value)
              })
            }
          />

        </div>


        <button
          className="route-button"
          onClick={findRoute}
          disabled={loading}
        >

          {loading
            ? "Finding route..."
            : "Find Safest Route"}

        </button>

      </section>


      {/* ==================================================
          ERROR
          ================================================== */}

      {error && (

        <div className="error-box">

          {error}

        </div>

      )}


      {/* ==================================================
          SUMMARY CARDS
          ================================================== */}

      {summary && (

        <section className="summary-grid">


          <div className="summary-card">

            <span>
              Route distance
            </span>

            <strong>
              {summary.distance_km} km
            </strong>

          </div>


          <div className="summary-card">

            <span>
              Dynamic cost
            </span>

            <strong>
              {summary.dynamic_cost}
            </strong>

          </div>


          <div className="summary-card">

            <span>
              Avg hazard
            </span>

            <strong>
              {(
                summary.average_hazard_probability *
                100
              ).toFixed(1)}
              %
            </strong>

          </div>


          <div className="summary-card">

            <span>
              Landslide
            </span>

            <strong>
              {(
                summary.average_landslide_probability *
                100
              ).toFixed(1)}
              %
            </strong>

          </div>


          <div className="summary-card">

            <span>
              Flood
            </span>

            <strong>
              {(
                summary.average_flood_probability *
                100
              ).toFixed(1)}
              %
            </strong>

          </div>


          <div className="summary-card">

            <span>
              Very high-risk segments
            </span>

            <strong>
              {summary.very_high_risk_segments}
            </strong>

          </div>

        </section>
      )}


      {/* ==================================================
          MAP
          ================================================== */}

      <section className="map-section">

        <MapContainer

          center={[
            26.25,
            92.15
          ]}

          zoom={9}

          className="map"

        >

          <TileLayer

            attribution='&copy; OpenStreetMap contributors'

            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"

          />


          {/* SOURCE */}

          <Marker
            position={[
              source.lat,
              source.lon
            ]}
          >

            <Popup>

              <strong>
                Source
              </strong>

              <br />

              Guwahati

            </Popup>

          </Marker>


          {/* DESTINATION */}

          <Marker
            position={[
              destination.lat,
              destination.lon
            ]}
          >

            <Popup>

              <strong>
                Destination
              </strong>

              <br />

              Nagaon

            </Popup>

          </Marker>


          {/* ROUTE */}

          {route && route.geometry && (

            <Polyline

              positions={
                route.geometry.coordinates.map(
                  ([lon, lat]) =>
                    [lat, lon]
                )
              }

              pathOptions={{
                color: getRiskColor(
                  summary?.average_hazard_probability || 0
                ),

                weight: 6,

                opacity: 0.85
              }}

            />

          )}


          {/* CHECKPOINTS */}

          {checkpoints.map(
            (checkpoint) => (

              <Marker

                key={
                  checkpoint.checkpoint_id
                }

                position={[
                  checkpoint.latitude,
                  checkpoint.longitude
                ]}

                icon={
                  createCheckpointIcon(
                    checkpoint.checkpoint_id
                      .replace("CP-", "")
                  )
                }

              >

                <Popup>

                  <div className="checkpoint-popup">

                    <h3>
                      {checkpoint.checkpoint_id}
                    </h3>

                    <p>
                      Distance:
                      {" "}
                      {checkpoint.distance_from_start_km}
                      {" km"}
                    </p>

                    <p>
                      Hazard:
                      {" "}
                      {(
                        checkpoint.p_hazard *
                        100
                      ).toFixed(1)}
                      %
                    </p>

                    <p>
                      Landslide:
                      {" "}
                      {(
                        checkpoint.p_landslide *
                        100
                      ).toFixed(1)}
                      %
                    </p>

                    <p>
                      Flood:
                      {" "}
                      {(
                        checkpoint.p_flood *
                        100
                      ).toFixed(1)}
                      %
                    </p>

                    <p>
                      Risk level:
                      {" "}
                      <strong>
                        {checkpoint.hazard_level}
                      </strong>
                    </p>

                  </div>

                </Popup>

              </Marker>

            )
          )}


          {/* FIT MAP TO ROUTE */}

          {route && (

            <FitRoute
              route={route}
            />

          )}

        </MapContainer>

      </section>


      {/* ==================================================
          RISK LEGEND
          ================================================== */}

      <section className="legend">

        <div className="legend-item">

          <span className="legend-color low"></span>

          Low

        </div>


        <div className="legend-item">

          <span className="legend-color moderate"></span>

          Moderate

        </div>


        <div className="legend-item">

          <span className="legend-color high"></span>

          High

        </div>


        <div className="legend-item">

          <span className="legend-color very-high"></span>

          Very High

        </div>

      </section>


    </div>
  );
}


export default App;